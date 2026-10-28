"""Prepared SM103 mHC compositions, raced as complete post/pre plus weighted norm calls.

FP32 projection weights retain BF16 high and low parts. The finalizer rounds the
collapsed residual to BF16 before computing its RMS, including the fused variant.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import (
    _NATIVE_JITS,
    attach_programs,
    compile_only_launches,
    launch_triton,
    load_programs,
)
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.fixtures.resources import candidate_inputs
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer

from .native_operator import native_contract, native_plan

SPLITS = 4
_BLOCK_T = 128
_BLOCK_K = 64
_CONFIG = triton.Config({}, num_warps=8, num_stages=3)
MIN_ROWS = 256
_SPLIT_WEIGHTS = {}


@dataclass(frozen=True)
class Query:
    rows: int
    hidden: int
    operation: str
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_eps: float
    norm_dtype: str
    has_low: bool
    source_identity: str

    def __post_init__(self):
        if (
            type(self.rows) is not int
            or not MIN_ROWS < self.rows <= (2**31 - 1) // 16
            or self.hidden not in (4096, 5120, 7168)
            or self.operation not in ("pre", "post_pre")
            or self.norm_dtype not in ("bfloat16", "float32")
            or type(self.has_low) is not bool
        ):
            raise ValueError("composed mHC requires prefill rows, four supported streams and weighted norm")
        if (
            any(not math.isfinite(value) or value <= 0 for value in (self.rms_eps, self.hc_eps, self.norm_eps))
            or type(self.sinkhorn_iters) is not int
            or not 0 < self.sinkhorn_iters <= 2**31 - 1
        ):
            raise ValueError("mHC epsilon values and Sinkhorn iterations must be positive")


@cache
def tactics(query):
    # (row tile, K tile, K splits, warps, stages, finalizer, finalizer block).
    # "fused_pair" finalizes two tokens per CTA with the same arithmetic as "fused";
    # it is measured only for hidden size 4096.
    # Two-dot stages 2/3 changed projections across identical captured replays;
    # the general FP32 path therefore races only the validated single-stage form.
    projections = ((128, 64, 4, 8, 3), (64, 128, 2, 4, 6), (64, 128, 4, 4, 4))
    return tuple(
        (*geometry[:4], 1 if query.has_low else geometry[4], mode, block)
        for geometry in projections
        if (4 * query.hidden) % (geometry[1] * geometry[2]) == 0
        for mode, block in (
            ("separate", 256),
            ("separate", 128),
            ("fused", 160),
            *((("fused_pair", 160),) if query.hidden == 4096 else ()),
            ("fused", 96),
            ("fused", 288),
        )
    )


CONTRACT = native_contract(component_id="flare.sm103_composed_mhc", query_type=Query, tactics=tactics)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _project(
    Residual,
    FnHigh,
    FnLow,
    Dot,
    Sqrsum,
    rows,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_LOW: tl.constexpr,
):
    tile, split = tl.program_id(0), tl.program_id(1)
    tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    live = tokens < rows
    mixes = tl.arange(0, 32)
    dot = tl.zeros((BLOCK_T, 32), tl.float32)
    squares = tl.zeros((BLOCK_T,), tl.float32)
    span = K // SPLITS
    for start in range(split * span, (split + 1) * span, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        values = tl.load(
            Residual + tokens.to(tl.int64)[:, None] * K + k[None, :],
            mask=live[:, None],
            other=0.0,
        )
        dot += tl.dot(values, tl.trans(tl.load(FnHigh + mixes[:, None] * K + k[None, :])))
        if HAS_LOW:
            dot += tl.dot(values, tl.trans(tl.load(FnLow + mixes[:, None] * K + k[None, :])))
        wide = values.to(tl.float32)
        squares += tl.sum(wide * wide, 1)
    tl.store(
        Dot + (split * rows + tokens.to(tl.int64))[:, None] * 24 + mixes[None, :],
        dot,
        mask=live[:, None] & (mixes[None, :] < 24),
    )
    tl.store(Sqrsum + split * rows + tokens, squares, mask=live)


# Declare these external JITs so in-process compilation evicts discovery placeholders.
_NATIVE_JITS.add(_project.fn)


def _split_weights(fn):
    """Share prepared halves while their original immutable weight tensor is alive."""
    key = (fn.device, fn.data_ptr(), tuple(fn.shape), tuple(fn.stride()), fn.dtype)
    entry = _SPLIT_WEIGHTS.get(key)
    if entry is not None and entry[0]() is not None:
        return entry[1]
    padded = torch.zeros((32, fn.shape[1]), dtype=torch.float32, device=fn.device)
    padded[:24] = fn
    high = padded.to(torch.bfloat16)
    low = (padded - high.float()).to(torch.bfloat16)
    result = (high, low, bool(torch.count_nonzero(low).item()))

    def release(reference):
        if (current := _SPLIT_WEIGHTS.get(key)) is not None and current[0] is reference:
            del _SPLIT_WEIGHTS[key]

    _SPLIT_WEIGHTS[key] = (weakref.ref(fn, release), result)
    return result


def _layout(query, tactic):
    splits, finalizer = tactic[2], tactic[5]
    fields = [
        ((splits, query.rows, 24), torch.float32),
        ((splits, query.rows), torch.float32),
    ]
    if finalizer == "separate":
        fields.append(((query.rows, query.hidden), torch.bfloat16))
    offset, result = 0, []
    for shape, dtype in fields:
        size = math.prod(shape) * dtype.itemsize
        result.append((offset, size, shape, dtype))
        offset += (size + 255) // 256 * 256
    return tuple(result), offset


def _views(workspace, query, tactic, start=0):
    """Carve the layout from `start` bytes into the one-dimensional byte workspace."""
    layout, nbytes = _layout(query, tactic)
    if (
        workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
        or start < 0
        or workspace.nbytes < start + nbytes
        or (workspace.data_ptr() + start) % 256 != 0
    ):
        raise ValueError("composed mHC scratch requires its selected byte capacity and 256-byte alignment")
    return tuple(
        workspace.narrow(0, start + offset, size).view(dtype).view(shape)
        for offset, size, shape, dtype in layout
    )


def _plan_offset(workspace, storage, nbytes):
    """Byte offset of the plan's scratch view inside the arena tensor that captured calls receive."""
    if storage.nbytes < nbytes:
        raise ValueError("composed mHC scratch plan is smaller than its selected layout")
    offset = storage.data_ptr() - workspace.data_ptr()
    if storage.dtype != torch.uint8 or offset < 0 or offset + storage.nbytes > workspace.nbytes:
        raise ValueError("composed mHC scratch plan lies outside its arena")
    return offset


def _project_options(query, tactic):
    block_t, block_k, splits, warps, stages = tactic[:5]
    return {
        "K": 4 * query.hidden,
        "SPLITS": splits,
        "BLOCK_T": block_t,
        "BLOCK_K": block_k,
        "HAS_LOW": query.has_low,
        "num_warps": warps,
        "num_stages": stages,
    }


def _norm_arguments(query, collapsed, weight, output):
    hidden, rows = query.hidden, query.rows
    args = (
        hidden,
        rows,
        collapsed,
        hidden,
        None,
        None,
        0,
        weight,
        output,
        hidden,
        None,
    )
    options = {
        "EPS": query.norm_eps,
        "BLOCK_SIZE": triton.next_power_of_2(hidden),
        "HAS_IN_SCALE": False,
        "HAS_OUT_SCALE": False,
        "HAS_OUTPUT": True,
        "HAS_RESIDUAL": False,
        "ROUND_BEFORE_WEIGHT": False,
        "num_warps": max(8, min(32, triton.next_power_of_2(hidden) // 256)),
    }
    return args, options


@program_cache(scope="preparation")
def compile_projection(payload, geometry, ordinal):
    query = Query(**dict(payload))
    # Finalizer choices share this exact projection executable.
    rows, hidden = query.rows, query.hidden
    residual = torch.empty((rows, 4 * hidden), dtype=torch.bfloat16, device="meta")
    high = torch.empty((32, 4 * hidden), dtype=torch.bfloat16, device="meta")
    low = torch.empty_like(high)
    dot = torch.empty((geometry[2], rows, 24), dtype=torch.float32, device="meta")
    squares = torch.empty((geometry[2], rows), dtype=torch.float32, device="meta")
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(
            _project.fn,
            (triton.cdiv(rows, geometry[0]), geometry[2], 1),
            residual,
            high,
            low,
            dot,
            squares,
            rows,
            **_project_options(query, geometry),
        )


@program_cache(scope="preparation")
def compile_norm(payload, ordinal):
    from flashinfer.triton.kernels.norm import rms_norm_kernel

    _NATIVE_JITS.add(rms_norm_kernel)
    query = Query(**dict(payload))
    collapsed = torch.empty((query.rows, query.hidden), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((query.hidden,), dtype=torch.float32, device="meta")
    args, options = _norm_arguments(query, collapsed, weight, torch.empty_like(collapsed))
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(rms_norm_kernel, (query.rows, 1, 1), *args, **options)


@program_cache(scope="preparation")
def compile_bf16_norm(payload, ordinal):
    import cutlass
    from b12x._lib.compiler import KernelCompileSpec
    from b12x._lib.compiler import compile as compile_cute
    from b12x._lib.utils.device import current_cuda_stream

    from .mhc_sm103_norm import BF16NormHost, pointer

    query = Query(**dict(payload))
    with torch.cuda.device(ordinal):
        return compile_cute(BF16NormHost(query.hidden, query.rows * query.hidden > 2**31 - 1), pointer(16), pointer(16), pointer(16),
            cutlass.Int64(1), cutlass.Float32(1e-6), current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("flare.mhc_bf16_norm", 2,
                (query.hidden, query.rows * query.hidden > 2**31 - 1, 103, True, query.source_identity)))


@cache
def _source_identity():
    # The nvcc mHC finalizer is not a Triton program: include its actual sources
    # and launch options in selection identity as well as the Python composition.
    import flashinfer.mhc
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.mhc import gen_mhc_module
    from flashinfer.norm import utils as norm_utils
    from flashinfer.norm.kernels import rmsnorm

    from . import mhc_sm103_norm

    spec = gen_mhc_module()
    paths = (
        *spec.sources,
        jit_env.FLASHINFER_CSRC_DIR / "tvm_ffi_utils.h",
        jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/layout.cuh",
        Path(flashinfer.mhc.__file__),
        Path(__file__), Path(mhc_sm103_norm.__file__), Path(rmsnorm.__file__), Path(norm_utils.__file__),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
        digest.update(b"\0")
    digest.update(
        repr(
            (
                spec.extra_cflags,
                spec.extra_cuda_cflags,
                spec.extra_ldflags,
                spec.extra_cuda_cflags_by_source,
            )
        ).encode()
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple
    projection: object
    norm: object | None

    def run(self, inputs, weights, outputs, workspace, scratch_offset=0):
        from flashinfer.mhc import (
            _mhc_post_impl,
            _mhc_pre_big_fuse_impl,
            _mhc_pre_big_fuse_with_norm_impl,
        )

        query, tactic = self.query, self.tactic
        residual, post, comb, normalized = outputs
        high, low, scale, base, norm_weight = weights
        views = _views(workspace, query, tactic, scratch_offset)
        dot, squares = views[:2]
        if query.operation == "post_pre":
            _mhc_post_impl(residual, *inputs)
        else:
            residual.copy_(inputs[0])
        self.projection[(triton.cdiv(query.rows, tactic[0]), tactic[2], 1)](
            residual, high, low, dot, squares, query.rows
        )
        if tactic[5] in ("fused", "fused_pair"):
            _mhc_pre_big_fuse_with_norm_impl(
                post,
                comb,
                normalized,
                dot,
                squares,
                residual,
                scale,
                base,
                norm_weight,
                query.rms_eps,
                query.hc_eps,
                query.hc_eps,
                2.0,
                query.sinkhorn_iters,
                query.norm_eps,
                tactic[2],
                tactic[6],
                2 if tactic[5] == "fused_pair" else 1,
            )
        else:
            collapsed = views[2]
            _mhc_pre_big_fuse_impl(
                post,
                comb,
                collapsed,
                dot,
                squares,
                residual,
                scale,
                base,
                4 * query.hidden,
                query.rms_eps,
                query.hc_eps,
                query.hc_eps,
                2.0,
                query.sinkhorn_iters,
                tactic[2],
                tactic[6],
            )
            if query.norm_dtype == "bfloat16":
                from .mhc_sm103_norm import launch

                launch(self.norm, collapsed, norm_weight, normalized, query.rows, query.norm_eps)
            else:
                args, _ = _norm_arguments(query, collapsed, norm_weight, normalized)
                self.norm[(query.rows, 1, 1)](*args)


@torch.library.custom_op(
    "flare::sm103_composed_mhc",
    mutates_args=("residual", "post", "comb", "normalized", "workspace"),
)
def _composed(
    inputs: list[torch.Tensor],
    high: torch.Tensor,
    low: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    normalized: torch.Tensor,
    workspace: torch.Tensor,
    scratch_offset: int,
    handle: int,
) -> None:
    plan = plan_from_handle(handle)
    state = require_prepared(plan, CONTRACT.component_id, residual.device)
    state.run(
        inputs,
        (high, low, scale, base, norm_weight),
        (residual, post, comb, normalized),
        workspace,
        scratch_offset,
    )


@_composed.register_fake
def _fake_composed(
    inputs,
    high,
    low,
    scale,
    base,
    norm_weight,
    residual,
    post,
    comb,
    normalized,
    workspace,
    scratch_offset,
    handle,
) -> None:
    pass


def make_plan(query, device):
    device = torch.device(device)
    payload = CONTRACT.encode_query(query)
    def compile_jobs(tactic, detected):
        jobs = [
            CompileJob.create(
                "ignition.components.mhc_sm103:compile_projection",
                payload,
                tactic[:5],
                detected.ordinal,
            )
        ]
        if tactic[5] == "separate":
            factory = "compile_norm" if query.norm_dtype == "float32" else "compile_bf16_norm"
            jobs.append(
                CompileJob.create(
                    f"ignition.components.mhc_sm103:{factory}",
                    payload,
                    detected.ordinal,
                )
            )
        return tuple(jobs)

    def materialize(tactic):
        from flashinfer.mhc import get_mhc_module

        projection = compile_projection(payload, tactic[:5], device.index)
        norm = None
        if tactic[5] == "separate":
            factory = compile_norm if query.norm_dtype == "float32" else compile_bf16_norm
            norm = factory(payload, device.index)
        # Loading rejects a compiled launch whose exact shared-memory requirement
        # exceeds this GPU's opt-in block limit, before the candidate is primed.
        load_programs((projection, norm))
        get_mhc_module()
        return attach_programs(State(query, tactic, projection, norm), projection, norm)

    return native_plan(
        contract=CONTRACT,
        query=query,
        device=device,
        scratch_bytes=lambda tactic: _layout(query, tactic)[1],
        materialize=materialize,
        compile_jobs=compile_jobs,
    )


class ComposedMHC:
    """One bound call; shared plan state contains only query, tactic and programs."""

    def __init__(
        self,
        operation,
        weights,
        scratch,
        *,
        inputs,
        session,
        name,
        workspace,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        norm_eps,
    ):
        from .mhc import _BenchmarkInputs

        rows, _, hidden = scratch.residual.shape
        self.operation, self.weights, self.scratch = operation, weights, scratch
        self.high, self.low, self.has_low = _split_weights(weights.fn)
        self.parameters = (
            self.high,
            self.low,
            weights.scale,
            weights.base,
            weights.norm,
        )
        self.query = query = Query(
            rows,
            hidden,
            operation,
            float(rms_eps),
            float(hc_eps),
            sinkhorn_iters,
            float(norm_eps),
            str(weights.norm.dtype).removeprefix("torch."),
            self.has_low,
            _source_identity(),
        )
        device = scratch.residual.device
        for tensor in (*inputs, *self.parameters, *scratch.bind_args().values()):
            if tensor.device != device or not tensor.is_contiguous():
                raise ValueError("composed mHC operands must be contiguous on one CUDA device")

        aligned = (
            (scratch.residual, 16),
            (scratch.normalized, 16),
            (weights.norm, 16),
            (self.high, 16),
            (self.low, 16),
        )
        if any(tensor.data_ptr() % alignment for tensor, alignment in aligned):
            raise ValueError("composed mHC requires aligned residual, normalized output and norm weights")

        self.plan = make_plan(query, device)
        benchmark_inputs = candidate_inputs(lambda: _BenchmarkInputs(scratch.residual, operation, False))

        def call(state, values, outputs, fixture=None):
            # Candidate scratch depends on its own split/finalizer choice, not the
            # default or previously selected plan. Bind before timing/capture.
            from b12x._lib.scratch import scratch_buffer_spec

            spec = scratch_buffer_spec(
                CONTRACT.component_id,
                nbytes=_layout(query, state.tactic)[1],
                device=device,
            )
            storage = planned_scratch((spec,), workspace)
            targets = tuple(outputs.bind_args().values())
            return PreparedCall(
                run=lambda: state.run(values, self.parameters, targets, storage),
                produce=None if fixture is None else fixture.produce,
                output=targets,
                owners=(*prime_owners(workspace, storage), fixture),
            )

        def benchmark(state):
            fixture = benchmark_inputs()
            return call(state, fixture.inputs, fixture.scratch, fixture)

        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=lambda state: call(state, inputs, scratch),
                    benchmark_call=benchmark,
                ),
            ),
        )

        def finalize():
            storage = planned_scratch(self.plan.scratch_specs(), workspace)
            # One canonical mutable operand exposes serialization across all arena users.
            # Independent dtype views would give AOTAutograd different alias bases.
            self.workspace = workspace.tensor if isinstance(workspace, ScratchArena) else storage
            # Captured calls index from the plan's own region, wherever the arena placed it.
            self.state = require_prepared(self.plan, CONTRACT.component_id, device)
            self.scratch_offset = _plan_offset(self.workspace, storage, _layout(query, self.state.tactic)[1])
            self.dot, self.squares, *collapsed = _views(self.workspace, query, self.state.tactic, self.scratch_offset)
            self.collapsed = collapsed[0] if collapsed else None

        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self, *inputs):
        _composed(
            list(inputs),
            *self.parameters,
            *self.scratch.bind_args().values(),
            self.workspace,
            self.scratch_offset,
            self.plan.handle,
        )
        return self.scratch.normalized

    def run(self, *inputs):
        self.state.run(
            inputs,
            self.parameters,
            tuple(self.scratch.bind_args().values()),
            self.workspace,
            self.scratch_offset,
        )
