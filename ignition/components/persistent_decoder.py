"""Prepared persistent WO, routed/shared experts, and lagged hyperconnections.

The CUDA implementation serves the DeepSeek geometry of four ranks, 5120
hidden features, 384 experts, top-six routing, and 576 expert features per
rank. Attention, prefix state and vocabulary projection retain their existing
components. All CUDA allocations belong to the package peer-storage provider.
"""

from dataclasses import dataclass, field
import itertools
from math import prod
import weakref

import torch
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.rebuild_manifest import declare_container_fields

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@dataclass(frozen=True)
class PersistentDecoderQuery:
    rows: int
    kernel_source: str
    torch_version: str
    compiler_identity: str
    cuda_version: str
    storage_contract: str = "peer-owned-flags-epoch-scratch-v1"
    shared_a16: int = 0
    world_size: int = 4
    hidden_size: int = 5120
    expert_features: int = 576


def decoder_query(*, rows, shared_a16=0):
    from flashinfer.jit.persistent_decoder import source_identity, toolchain_identity

    return PersistentDecoderQuery(
        rows=rows,
        shared_a16=shared_a16,
        kernel_source=source_identity(),
        compiler_identity=toolchain_identity(),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda),
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
            or rep.source_format != "fp4_e8m0_k32"
            or rep.w13_layout != "w13"
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


@torch.library.custom_op(
    "flare::persistent_decoder", mutates_args=("io", "scratch", "peers")
)
def _execute(
    io: list[torch.Tensor],
    weights: list[torch.Tensor],
    scratch: list[torch.Tensor],
    peers: list[torch.Tensor],
    handle: int,
) -> None:
    owner = _OWNERS[handle]
    owner.channel.module.run(
        io, weights, scratch, peers, owner.channel.rank, owner.ctas, owner.shared_a16
    )


@_execute.register_fake
def _fake(io, weights, scratch, peers, handle):
    pass


def _validate_operands(io, shapes, *, operation):
    if len(io) != len(shapes):
        raise ValueError(
            f"{operation} requires {len(shapes)} tensor operands, got {len(io)}"
        )
    device = io[0].device
    if device.type != "cuda":
        raise ValueError(f"{operation} requires CUDA tensors")
    for tensor, (shape, dtype, role) in zip(io, shapes, strict=True):
        expected = tensor.numel() if shape is None else prod(shape)
        if (
            tensor.device != device
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tensor.numel() != expected
            or (shape is not None and tuple(tensor.shape) != shape)
        ):
            raise ValueError(
                f"{operation} {role} must be contiguous {dtype} {shape} on {device}"
            )


def _validate_tail(io):
    if len(io) != 20:
        raise ValueError("persistent decoder tail requires twenty tensor operands")
    rows = io[4].shape[0]
    if not 0 < rows <= 128 or io[0].numel() != rows * 8192:
        raise ValueError(
            "persistent decoder tail requires one through 128 rows with sixteen 512-wide heads"
        )
    bf, fp, integer, long = torch.bfloat16, torch.float32, torch.int32, torch.int64
    _validate_operands(
        io,
        (
            (None, bf, "attended values"),
            ((rows,), long, "positions"),
            ((rows,), long, "token IDs"),
            (None, fp, "rotary angles"),
            ((rows, 4, 5120), bf, "attention residual"),
            ((rows, 4), fp, "attention output gates"),
            ((rows, 4, 4), fp, "attention residual map"),
            ((rows, 4), fp, "attention input gates"),
            ((rows, 5120), bf, "FFN output"),
            ((rows, 4, 5120), bf, "FFN residual"),
            ((rows, 4), fp, "FFN output gates"),
            ((rows, 4, 4), fp, "FFN residual map"),
            ((rows, 4), fp, "FFN input gates"),
            ((rows, 384), fp, "router logits"),
            ((rows, 6), integer, "expert IDs"),
            ((rows, 6), fp, "route weights"),
            ((rows, 5120), bf, "routed output"),
            ((rows, 5120), bf, "shared output"),
            ((rows, 1152), bf, "shared gate/up boundary"),
            ((rows, 576), bf, "shared activation boundary"),
        ),
        operation="persistent decoder",
    )


@dataclass(eq=False)
class PersistentDecoderCall:
    """A bound persistent tail whose explicit tensor owners survive capture."""

    io: list
    weights: list
    scratch: list
    peers: list
    channel: object
    ctas: int
    shared_a16: int
    owners: tuple
    handle: int = field(init=False)

    def __post_init__(self):
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    @classmethod
    def bind(cls, io, weights, channel, *, workers, shared_a16):
        _validate_tail(io)
        return cls(
            list(io),
            list(weights.tensors),
            channel.scratch,
            channel.peers,
            channel,
            channel.ctas[workers],
            shared_a16,
            (weights, channel),
        )

    def __call__(self):
        _execute(self.io, self.weights, self.scratch, self.peers, self.handle)
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
    handle: int,
) -> None:
    owner = _OWNERS[handle]
    owner.channel.module.prologue(io, weights, scratch, owner.ctas)


@_prologue.register_fake
def _fake_prologue(io, weights, scratch, handle):
    pass


@dataclass(eq=False)
class PersistentPrologueCall:
    """A bound prologue using the channel's explicitly retained scratch."""

    io: list
    weights: list
    scratch: list
    ctas: int
    owners: tuple
    channel: object
    handle: int = field(init=False)

    def __post_init__(self):
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    @classmethod
    def bind(cls, io, weights, channel):
        return cls(
            list(io),
            list(weights.tensors),
            channel.scratch,
            channel.prologue_ctas,
            (weights, channel),
            channel,
        )

    def __call__(self):
        _prologue(self.io, self.weights, self.scratch, self.handle)
        return self.io[8]


# Prepared native operation owners contain runtime state as well as weight views.
declare_container_fields(__name__ + ".PersistentDecoderWeights", ("tensors",))
declare_container_fields(__name__ + ".PersistentPrologueWeights", ("tensors",))
