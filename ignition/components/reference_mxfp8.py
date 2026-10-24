"""Reference FP8 attention projections prepared from CUDA and Triton sources.

The reference recipe keeps the normal BF16 normalization boundary and uses
E4M3 activations with per-32 UE8M0 scales and an absmax floor of 1e-4.
At one through eight rows it uses a source-built MoET GEMV; larger capacities
use the prepared b12x FP8 GEMM with the same reference quantizer.
"""

from dataclasses import replace

import torch
from b12x.gemm import blockscaled
from b12x.gemm.blockscaled._a16 import _weight_parts
from b12x.gemm.reference_mxfp8 import projection_module, quantizer, norm_quantizer
from b12x.preparation import PreparedCall
from b12x._lib.compile_pool import CompileJob
from b12x.preparation.types import plan_from_handle

from ignition.preparation.scratch import scratch_storage, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer


def _run_projection(state, source, values, scales, out, scratch, quantized, alpha):
    q = state.query
    sf_start, alpha_start, partial_start, needed = state.offsets
    count = source.shape[0]
    if quantized:
        packed, scale_bytes = quantized
    else:
        packed = (
            scratch[: count * q.in_features]
            .view(torch.float8_e4m3fn)
            .view(count, q.in_features)
        )
        size = -(-count // 128) * -(-(q.in_features // 32) // 4) * 512
        scale_bytes = scratch[sf_start : sf_start + size]
        quantizer(count, q.in_features)(source, packed, scale_bytes)
    if count <= 8:
        projection_module().projection_out(packed, scale_bytes, values, scales, out, 1)
    else:
        from b12x._lib.intrinsics.reference import as_grouped_scale_view_mx

        state.dense.run(
            (
                packed.view(count, q.in_features, 1),
                as_grouped_scale_view_mx(scale_bytes.view(1, -1), count, q.in_features),
            ),
            (
                values.view(q.out_features, q.in_features, 1),
                scales.view(torch.float8_e8m0fnu),
            ),
            out=out.view(count, q.out_features, 1),
            alpha=alpha,
            split_k_workspace=scratch[partial_start:needed].view(torch.float32),
        )


@torch.library.custom_op(
    "flare::reference_mxfp8_projection", mutates_args=("out", "scratch")
)
def _project(
    source: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    scratch: torch.Tensor,
    quantized: list[torch.Tensor],
    alpha: torch.Tensor,
    handle: int,
) -> None:
    _run_projection(
        plan_from_handle(handle).prepared.state,
        source,
        values,
        scales,
        out,
        scratch,
        quantized,
        alpha,
    )


@_project.register_fake
def _project_fake(source, values, scales, out, scratch, quantized, alpha, handle):
    pass


class ReferenceMXFP8Projection:
    def __init__(
        self, source, weight, output, *, session, name, workspace, quantized=()
    ):
        from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

        if not isinstance(weight, MXFP8LinearWeight) or output.dtype != torch.bfloat16:
            raise ValueError(
                "reference attention projections require checkpoint MXFP8 weights and BF16 output"
            )
        if (
            not source.is_contiguous()
            or weight.in_features != weight.padded_in_features
        ):
            raise ValueError(
                "reference attention projections require contiguous, unpadded input"
            )
        self.weight, self.output, self.workspace = weight, output, workspace
        self.prequantized = tuple(quantized)
        values, scales = _weight_parts(weight)[:2]
        # Opaque compiled calls carry UE8M0 storage as unchanged bytes.
        # The typed dense consumer restores its numeric view at launch.
        self.parts = values, scales.view(torch.uint8)
        for tensor in self.parts:
            record_numeric_tensor(tensor)
        projection_module()
        self.alpha = torch.ones(1, dtype=torch.float32, device=source.device)
        record_numeric_tensor(self.alpha)
        self.quantize = quantizer(source.shape[0], source.shape[-1])
        query = blockscaled.query_from_call(
            source,
            weight,
            out=output,
            activation_mode="quantized",
            expected_m=source.shape[0],
        )
        query = replace(query, workspace_form="provided", workspace_nbytes=None)
        # The source GEMV is a fixed recipe choice. Native GEMM knobs do not
        # describe it; reserve the full canonical search for wider capacities.
        override = (
            blockscaled.BlockscaledConfig(mode="quantized")
            if source.shape[0] <= 8
            else None
        )
        base = blockscaled.plan(query, override=override)
        self.plan = replace(
            base,
            _compile_jobs=lambda config, device: (
                *base._compile_jobs(config, device),
                CompileJob.create(
                    "b12x.gemm.reference_mxfp8:compile_quantizer",
                    source.shape[0],
                    source.shape[-1],
                    CompileJob.DEVICE_ORDINAL,
                ),
            ),
        )

        def storage(state):
            return scratch_storage(
                workspace, nbytes=state.required_workspace, device=source.device
            )

        def prepare(state):
            scratch = storage(state)
            # Tuning must receive actual activations and prequantized values.
            seed = torch.randn_like(source) * 0.125

            def produce():
                source.copy_(seed)
                if self.prequantized:
                    self.quantize(source, *self.prequantized)

            return PreparedCall(
                run=lambda: _run_projection(
                    state,
                    source,
                    *self.parts,
                    output,
                    scratch,
                    list(self.prequantized),
                    self.alpha,
                ),
                output=output,
                produce=produce,
                owners=(self, seed, *prime_owners(workspace, scratch)),
            )

        declare(
            session,
            (
                self.plan.request(
                    name=name, prepare_call=prepare, benchmark_call=prepare
                ),
            ),
        )
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        state = self.plan.prepared.state
        if self.output.shape[0] > 8 and state.dense.lowering.policy.split_k_slices != 1:
            raise ValueError(
                "reference attention fallback requires single-partition FP32 accumulation"
            )
        self.scratch = scratch_storage(
            self.workspace, nbytes=state.required_workspace, device=self.output.device
        )

    def __call__(self, source):
        _project(
            source,
            *self.parts,
            self.output,
            self.scratch,
            list(self.prequantized),
            self.alpha,
            self.plan.handle,
        )
        return self.output


@torch.library.custom_op(
    "flare::reference_query_norm_quant",
    mutates_args=("query", "kv", "quantized", "scales", "scale_rows"),
)
def _norm_quant(
    projected: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    scale_rows: torch.Tensor,
) -> None:
    norm_quantizer(projected.shape[0], qw.dtype, pdl=True)(
        projected, qw, kw, query, kv, quantized, scales, scale_rows
    )


@_norm_quant.register_fake
def _norm_quant_fake(projected, qw, kw, query, kv, quantized, scales, scale_rows):
    pass


class ReferenceQueryNormQuant:
    def __init__(self, projected, weights, query, kv, *, buffers):
        rows = projected.shape[0]
        if (
            projected.shape[1] != 1792
            or query.shape != (rows, 1280)
            or kv.shape != (rows, 512)
        ):
            raise ValueError(
                "reference query normalization requires the 1280+512 attention projection"
            )
        if weights[0].dtype != weights[1].dtype:
            raise ValueError(
                "reference Q/KV normalization requires matching affine dtypes"
            )
        self.weights = weights
        for weight in weights:
            record_numeric_tensor(weight)
        self.query, self.kv = query, kv
        value_bytes, scale_bytes = rows * 1280, -(-rows // 128) * 10 * 512
        if buffers is None:
            raise ValueError(
                "reference query normalization requires pooled caller storage"
            )
        self.backing = buffers.full(
            value_bytes + scale_bytes + rows * 40, 127, torch.uint8
        )
        self.quantized = (
            self.backing[:value_bytes].view(torch.float8_e4m3fn).view(rows, 1280)
        )
        self.scales = self.backing[value_bytes : value_bytes + scale_bytes]
        self.scales.fill_(127)
        self.scale_rows = self.backing[value_bytes + scale_bytes :].view(rows, 40)
        norm_quantizer(rows, weights[0].dtype, pdl=True)

    def __call__(self, projected):
        _norm_quant(
            projected,
            *self.weights,
            self.query,
            self.kv,
            self.quantized,
            self.scales,
            self.scale_rows,
        )


@torch.library.custom_op(
    "flare::start_reference_projection", mutates_args=("out", "scratch")
)
def _start_reference(
    source: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    scratch: torch.Tensor,
    quantized: list[torch.Tensor],
    alpha: torch.Tensor,
    handle: int,
    stream: int,
) -> None:
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        _run_projection(
            plan_from_handle(handle).prepared.state,
            source,
            values,
            scales,
            out,
            scratch,
            quantized,
            alpha,
        )


@_start_reference.register_fake
def _start_reference_fake(
    source, values, scales, out, scratch, quantized, alpha, handle, stream
):
    pass


@torch.library.custom_op(
    "flare::join_reference_projection", mutates_args=("outputs", "workspace")
)
def _join_reference(
    source: torch.Tensor,
    dependency: torch.Tensor,
    quantized: list[torch.Tensor],
    outputs: list[torch.Tensor],
    workspace: torch.Tensor,
    stream: int,
) -> None:
    # These read operands retain the actual prequantized input storage until
    # side completion, including snapshots introduced by storage lowering.
    torch.cuda.current_stream(source.device).wait_stream(
        torch.cuda.ExternalStream(stream, device=source.device)
    )


@_join_reference.register_fake
def _join_reference_fake(source, dependency, quantized, outputs, workspace, stream):
    pass


class ReferenceProjectionChain:
    """Keep the query projection's existing fork/join and disjoint scratch."""

    def __init__(self, projection, *, parallel_plans, session, workspace):
        self.projection, self.parallel_plans, self.arena = (
            projection,
            tuple(parallel_plans),
            workspace,
        )
        self.outputs = [projection.output]
        for plan in self.parallel_plans:
            workspace.register_parallel(plan, projection.plan)
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        main = max(
            plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans
        )
        side = self.projection.plan.memory_requirements().scratch_nbytes
        _, self.workspace = self.arena.views(
            ((main,), torch.uint8), ((side,), torch.uint8)
        )

    def start(self, source):
        p = self.projection
        _start_reference(
            source,
            *p.parts,
            p.output,
            self.workspace,
            list(p.prequantized),
            p.alpha,
            p.plan.handle,
            self.stream_handle,
        )

    def finish(self, source, dependency):
        _join_reference(
            source,
            dependency,
            list(self.projection.prequantized),
            self.outputs,
            self.workspace,
            self.stream_handle,
        )
        return self.outputs[-1]
