"""Prepared persistent WO, routed/shared experts, and lagged hyperconnections.

The CUDA implementation serves the DeepSeek geometry of four ranks, 5120
hidden features, 384 experts, top-six routing, and 576 expert features per
rank. Attention, prefix state and vocabulary projection retain their existing
components. All CUDA allocations belong to the package peer-storage provider.
"""

from dataclasses import dataclass

import torch
from b12x.preparation import (
    CollectiveRequirement,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    PreparedCall,
)

from ignition.components.native_operator import native_contract
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class PersistentDecoderQuery:
    rows: int
    shared_a16: int = 0
    world_size: int = 4
    hidden_size: int = 5120
    expert_features: int = 576


def _tactics(query):
    if (
        not 0 < query.rows <= 128
        or query.world_size != 4
        or query.hidden_size != 5120
        or query.expert_features != 576
        or query.shared_a16 not in (0, 1, 2, 3)
        or query.shared_a16
        and query.rows > 8
    ):
        raise ValueError(
            "persistent decoder requires TP4, 1..128 rows, H5120 and expert width576"
        )
    return (1, 2)


CONTRACT = native_contract(
    component_id="flare.persistent_decoder.tp4",
    query_type=PersistentDecoderQuery,
    tactics=_tactics,
)


def _stages(weight):
    """Pack 64-output-row weight tiles for asynchronous persistent consumers."""
    raw = weight.contiguous().view(torch.uint8)
    raw = torch.nn.functional.pad(raw, (0, -raw.shape[1] % 128))
    return (
        raw.reshape(raw.shape[0] // 64, 4, 2, 8, -1, 4, 4, 8)
        .permute(0, 4, 5, 1, 3, 6, 2, 7)
        .contiguous()
        .view(-1)
    )


def _interleave(weight, block):
    return (
        weight.reshape(2, -1, block, *weight.shape[1:])
        .transpose(0, 1)
        .reshape(weight.shape)
        .contiguous()
    )


def _dense_rows(packed):
    values = packed.values
    if values.ndim == 3:
        return values.permute(2, 0, 1).contiguous().view(-1, values.shape[1])
    return values


def _scale_rows(packed):
    scales = packed.scale_rows.view(torch.uint8)
    return scales.reshape(-1, scales.shape[-1])[::32].contiguous()


@dataclass(frozen=True)
class PersistentDecoderWeights:
    """Persistent dense tiles and aliases of the prepared compact-N64 experts.

    Dense tiles coexist with the ordinary prefill weights. Expert values and
    scales alias the existing b12x representation without another expert copy.
    """

    tensors: tuple
    dense_nbytes: int
    expert_owner: object

    @classmethod
    def pack(cls, layer):
        output, experts, residual = (
            layer.attention["output"],
            layer.experts,
            layer.residual,
        )
        if (output.groups, output.group_width, output.rank, output.hidden) != (
            2,
            4096,
            1024,
            5120,
        ):
            raise ValueError(
                "persistent decoder output weights require two 4096-to-1024 groups"
            )
        rep = experts.experts._impl.representation_for("w4a8_mx")
        if (
            rep is None
            or not rep.n64_repack
            or rep.intermediate_size != 576
            or rep.num_experts != 384
            or rep.hidden_size != 5120
        ):
            raise ValueError(
                "persistent decoder requires prepared compact-N64 MXFP4 experts"
            )
        shared = experts.shared
        if experts.router.dtype != torch.bfloat16:
            raise ValueError(
                "persistent decoder router preserves BF16 checkpoint weights"
            )
        gate_up, down = shared.gate_up.weight, shared.down.weight
        dense = (
            _stages(_dense_rows(output.wo_a)),
            _scale_rows(output.wo_a),
            _stages(_dense_rows(output.wo_b)),
            _scale_rows(output.wo_b),
            _stages(experts.router),
            experts.correction.float().contiguous(),
            torch.empty(0, dtype=torch.float32, device=experts.router.device)
            if experts.vision_correction is None
            else experts.vision_correction.float().contiguous(),
            _stages(_interleave(_dense_rows(gate_up), 32)),
            _interleave(_scale_rows(gate_up), 1),
            _stages(_dense_rows(down)),
            torch.nn.functional.pad(
                _scale_rows(down), (0, 20 - _scale_rows(down).shape[1])
            ),
        )
        tail = (
            residual["hc_ffn_fn"].float().contiguous(),
            residual["hc_ffn_scale"].float().contiguous(),
            residual["hc_ffn_base"].float().contiguous(),
            residual["ffn_norm.weight"].contiguous(),
        )
        tensors = (*dense, rep.w13_rp, rep.w13_sfb, rep.w2_rp, rep.w2_sfb, *tail)
        for tensor in tensors:
            record_numeric_tensor(tensor)
        return cls(tensors, sum(tensor.nbytes for tensor in (*dense, *tail)), rep)


@dataclass(frozen=True)
class _State:
    channel: object
    blocks_per_sm: int
    shared_a16: int

    def run(self, io, weights):
        self.channel.run(
            io, weights, blocks_per_sm=self.blocks_per_sm, shared_a16=self.shared_a16
        )


@torch.library.custom_op(
    "flare::persistent_decoder", mutates_args=("io", "scratch", "peers")
)
def _execute(
    io: list[torch.Tensor],
    weights: list[torch.Tensor],
    scratch: list[torch.Tensor],
    peers: list[torch.Tensor],
    rank: int,
    ctas: int,
    shared_a16: int = 0,
) -> None:
    from flashinfer.experimental.persistent_decoder import module

    module().run(io, weights, scratch, peers, rank, ctas, shared_a16)


@_execute.register_fake
def _fake(io, weights, scratch, peers, rank, ctas, shared_a16=0):
    pass


class PreparedPersistentDecoder:
    """Declare one complete decoder candidate and bind selected resident workers.

    ``channel_factory`` creates or returns the rank's shared channel during
    plan materialization, after the caller has entered collective preparation.
    ``benchmark_call`` must supply the complete layer's real input-production
    and correctness contract when racing this candidate against separated work.
    """

    def __init__(
        self,
        io,
        weights,
        *,
        channel_factory,
        session,
        coordinator,
        group,
        name,
        benchmark_call=None,
        shared_a16=0,
    ):
        import torch.distributed as dist

        self.io, self.weights = list(io), list(weights.tensors)
        query = PersistentDecoderQuery(rows=io[4].shape[0], shared_a16=shared_a16)
        self.shared_a16 = shared_a16
        self.plan = Plan(
            contract=CONTRACT,
            query=query,
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(
                persistent=(
                    PersistentMemory(
                        ("persistent_decoder_channel", channel_factory), 39795456
                    ),
                    PersistentMemory(
                        ("persistent_decoder_dense", id(weights)),
                        weights.dense_nbytes,
                        weights.dense_nbytes,
                    ),
                )
            ),
            _materialize=lambda selection, device: _State(
                channel_factory(), selection.config.tactic, shared_a16
            ),
        )
        ranks = tuple(dist.get_process_group_ranks(group))
        request = self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(self.io, self.weights),
                output=tuple(self.io[8:]),
                owners=(state.channel, weights, *self.io),
            ),
            benchmark_call=benchmark_call,
            collective=CollectiveRequirement(key=name, ranks=ranks),
        )
        declare(session, (request,), coordinator=coordinator)
        register_finalizer(session, self.rebind)

    def rebind(self):
        state = self.plan.prepared.state
        self.channel = state.channel
        self.scratch, self.peers, self.rank = (
            self.channel.scratch,
            self.channel.peers,
            self.channel.rank,
        )
        self.ctas = self.channel.ctas[state.blocks_per_sm]

    def __call__(self):
        _execute(
            self.io,
            self.weights,
            self.scratch,
            self.peers,
            self.rank,
            self.ctas,
            self.shared_a16,
        )
        return self.io[8]


@dataclass(frozen=True)
class PersistentPrologueWeights:
    """Affine tensors and row-packed FP8 query/KV projections for a prologue."""

    tensors: tuple
    packed_nbytes: int

    @classmethod
    def pack(cls, layer):
        attention, residual = layer.attention, layer.residual

        def projection(weight):
            packed = weight.weight
            values = _dense_rows(packed).contiguous()
            n, k = values.shape
            scales = packed.scale_rows.view(torch.uint8).reshape(n, k // 32)
            scales = (
                scales.reshape(n // 8, 4, 2, k // 64, 2)
                .permute(0, 3, 1, 4, 2)
                .contiguous()
            )
            return values.view(torch.uint8), scales

        query_kv = projection(attention["qkv_a"])
        query = projection(attention["q_b"])
        affine = (
            residual["attn_norm.weight"],
            attention["q_norm"],
            attention["kv_norm"],
        )
        if any(t.dtype != torch.bfloat16 for t in affine):
            raise ValueError(
                "persistent prologue requires BF16 affine checkpoint weights"
            )
        tensors = (
            residual["hc_attn_fn"].float().contiguous(),
            affine[0],
            *query_kv,
            affine[1],
            affine[2],
            *query,
            residual["hc_attn_scale"].float().contiguous(),
            residual["hc_attn_base"].float().contiguous(),
        )
        for tensor in tensors:
            record_numeric_tensor(tensor)
        return cls(tensors, query_kv[1].nbytes + query[1].nbytes)


@torch.library.custom_op(
    "flare::persistent_decoder_prologue", mutates_args=("io", "scratch")
)
def _prologue(
    io: list[torch.Tensor],
    weights: list[torch.Tensor],
    scratch: list[torch.Tensor],
    ctas: int,
) -> None:
    from flashinfer.experimental.persistent_decoder import module

    module().prologue(io, weights, scratch, ctas)


@_prologue.register_fake
def _fake_prologue(io, weights, scratch, ctas):
    pass


class PreparedPersistentPrologue:
    """Fuse lagged mHC, input/query projections, normalization and rotary work.

    Input angles are per-row interleaved FP32 cosine/sine pairs from the
    existing prepared rotary-angle producer. KV cache writes retain their
    existing component; this operator produces the rotated BF16 KV values.
    The recipe must exclude Engram-bearing layers and initial residual-only
    boundaries, which require their existing preparation composition.
    """

    def __init__(
        self, io, weights, *, channel_factory, session, name, coordinator, group
    ):
        import torch.distributed as dist

        self.io, self.weights = list(io), list(weights.tensors)
        rows = io[1].shape[0]
        if not 0 < rows <= 8:
            raise ValueError("persistent prologue requires one through eight rows")
        contract = native_contract(
            component_id="flare.persistent_decoder.prologue",
            query_type=PersistentDecoderQuery,
            tactics=lambda query: (0,),
        )
        self.plan = Plan(
            contract=contract,
            query=PersistentDecoderQuery(rows=rows),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(
                persistent=(
                    PersistentMemory(
                        ("persistent_decoder_channel", channel_factory), 39795456
                    ),
                    PersistentMemory(
                        ("persistent_prologue_weights", id(weights)),
                        weights.packed_nbytes,
                        weights.packed_nbytes,
                    ),
                )
            ),
            _materialize=lambda selection, device: channel_factory(),
        )
        ranks = tuple(dist.get_process_group_ranks(group))
        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=lambda channel: PreparedCall(
                        run=lambda: channel.prologue(self.io, self.weights),
                        output=tuple(self.io[7:]),
                        owners=(channel, weights, *self.io),
                    ),
                    collective=CollectiveRequirement(key=name, ranks=ranks),
                ),
            ),
            coordinator=coordinator,
        )
        register_finalizer(session, self.rebind)

    def rebind(self):
        channel = self.plan.prepared.state
        self.channel, self.scratch, self.ctas = (
            channel,
            channel.scratch,
            channel.prologue_ctas,
        )

    def __call__(self):
        _prologue(self.io, self.weights, self.scratch, self.ctas)
        return self.io[8]
