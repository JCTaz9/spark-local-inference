"""mHC prefill on SM103 through FlashInfer's kernels against b12x's references."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

# GLM-5.3 and DeepSeek V4 both mix four 4096-wide streams; V4 normalizes with eps 1e-20.
MODELS = {
    "glm53": (4096, {"rms_eps": 1e-6, "hc_eps": 1e-6, "sinkhorn_iters": 20, "norm_eps": 1e-5}),
    "deepseek4": (4096, {"rms_eps": 1e-20, "hc_eps": 1e-6, "sinkhorn_iters": 20, "norm_eps": 1e-20}),
}


@pytest.fixture(scope="module")
def weight_owners():
    # Prepared models retain original weights while the split-weight cache is live.
    return []


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("exact_bf16_fn", [False, True])
def test_composed_mhc_matches_reference(operation, model, norm_dtype, exact_bf16_fn, weight_owners):
    from b12x.preparation import PreparationSession
    from b12x.testing.mhc import post_reference, pre_reference

    from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
    from ignition.preparation.scratch import ScratchArena

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("composed mHC serves SM103")
    torch.manual_seed(0)
    rows, (hidden, eps) = 1024, MODELS[model]
    fn = torch.randn(24, 4 * hidden, device="cuda") * 0.02
    if exact_bf16_fn:
        fn = fn.bfloat16().float()
    weights = MHCWeights(fn, torch.rand(3, device="cuda") * 0.5,
                         torch.randn(24, device="cuda") * 0.3,
                         (torch.rand(hidden, device="cuda") + 0.5).to(norm_dtype))
    weight_owners.append(weights)
    residual = torch.randn(rows, 4, hidden, device="cuda").to(torch.bfloat16)
    inputs = (residual,)
    if operation == "post_pre":
        source = (torch.randn(rows, hidden, device="cuda") * 0.5).to(torch.bfloat16)
        post = torch.rand(rows, 4, device="cuda") * 2
        comb = torch.softmax(torch.randn(rows, 4, 4, device="cuda"), -1)
        inputs = (source, residual, post, comb)
    scratch = MHCScratch.allocate(rows, hidden, "cuda")
    arena = ScratchArena("cuda:0")
    with PreparationSession(device=torch.device("cuda"), autotune=True, compile_workers=0) as session:
        operator = PreparedMHC(inputs, weights, scratch, session=session, name=operation, workspace=arena, **eps)
        session.freeze()
        assert operator.composed is not None
        assert operator.composed.has_low is not exact_bf16_fn
        graph = torch.cuda.CUDAGraph()
        operator(*inputs)
        with torch.cuda.graph(graph):
            operator(*inputs)
        scratch.normalized.zero_()
        graph.replay()
        torch.cuda.synchronize()

        mixed = residual if operation == "pre" else post_reference(*inputs)
        collapsed, post_mix, comb_mix = pre_reference(mixed, weights.fn, weights.scale, weights.base,
            rms_eps=eps["rms_eps"], hc_eps=eps["hc_eps"], sinkhorn_iters=eps["sinkhorn_iters"])
        wide = collapsed.float()
        normalized = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps["norm_eps"])
                      * weights.norm.float()).bfloat16()

        def relative(actual, expected):
            return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()

        assert relative(scratch.residual, mixed) < 1e-3
        # The projection reproduces FP32 weights to 16 bits (b12x uses TF32, 10).
        assert relative(scratch.post, post_mix) < 1e-3
        assert relative(scratch.comb, comb_mix) < 1e-3
        assert relative(scratch.normalized, normalized) < 5e-3

        if exact_bf16_fn:
            from ignition.components.mhc_sm103 import _project, _project_options

            composed = operator.composed
            tactic = composed.state.tactic
            args = (scratch.residual, composed.high, composed.low, composed.dot, composed.squares, rows)
            grid = ((rows + tactic[0] - 1) // tactic[0], tactic[2], 1)
            # Compare the exact same reduction geometry with a safe single-stage
            # pipeline: the forced two-dot branch must not reintroduce stages2/3.
            options = _project_options(composed.query, tactic) | {"num_stages": 1}
            one = _project.fn.warmup(*args, grid=grid, **options)
            full = _project.fn.warmup(*args, grid=grid, **(options | {"HAS_LOW": True}))
            assert full.hash != one.hash
            one[grid](*args)
            saved_dot, saved_squares = composed.dot.clone(), composed.squares.clone()
            full[grid](*args)
            torch.cuda.synchronize()
            torch.testing.assert_close(composed.dot, saved_dot, rtol=0, atol=0)
            torch.testing.assert_close(composed.squares, saved_squares, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_full_fp32_mhc_poisoned_replays_are_exact(hidden, operation, norm_dtype, weight_owners):
    from b12x.preparation import PreparationSession
    from b12x.testing.mhc import post_reference, pre_reference

    from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("composed mHC serves SM103")
    torch.manual_seed(8103)
    rows = 4096
    eps = 1e-5 if hidden == 4096 else 1e-20
    weights = MHCWeights(torch.randn(24, 4 * hidden, device="cuda") * .02,
        torch.rand(3, device="cuda") * .5, torch.randn(24, device="cuda") * .3,
        (torch.rand(hidden, device="cuda") + .5).to(norm_dtype))
    weight_owners.append(weights)
    residual = (torch.randn(rows, 4, hidden, device="cuda") * .1).bfloat16()
    inputs = (residual,)
    if operation == "post_pre":
        inputs = ((torch.randn(rows, hidden, device="cuda") * .1).bfloat16(), residual,
                  torch.rand(rows, 4, device="cuda") * 2,
                  torch.softmax(torch.randn(rows, 4, 4, device="cuda"), -1))
    scratch = MHCScratch.allocate(rows, hidden, "cuda")
    with PreparationSession(device=torch.device("cuda"), autotune=False, compile_workers=0) as session:
        prepared = PreparedMHC(inputs, weights, scratch, session=session, name=operation,
            workspace=None, rms_eps=eps, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=eps)
        session.freeze()
        composed = prepared.composed
        assert composed is not None and composed.has_low
        assert composed.state.tactic[4] == 1
        prepared(*inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepared(*inputs)
        graph.replay()
        torch.cuda.synchronize()
        outputs = (*scratch.bind_args().values(), composed.dot, composed.squares)
        expected_replay = tuple(value.clone() for value in outputs)
        allocated = torch.cuda.memory_allocated()
        for _ in range(8):
            temporaries = () if composed.collapsed is None else (composed.collapsed,)
            for value in (*outputs, *temporaries):
                value.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            for got, want in zip(outputs, expected_replay, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

        mixed = residual if operation == "pre" else post_reference(*inputs)
        collapsed, post, comb = pre_reference(mixed, weights.fn, weights.scale, weights.base,
            rms_eps=eps, hc_eps=1e-6, sinkhorn_iters=20)
        wide = collapsed.float()
        normalized = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps)
                      * weights.norm.float()).bfloat16()
        for got, want, bound in zip(scratch.bind_args().values(), (mixed, post, comb, normalized),
                                   (1e-3, 1e-3, 1e-3, 5e-3), strict=True):
            assert torch.isfinite(got).all()
            assert ((got.float() - want.float()).norm() / want.float().norm()).item() < bound


@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_low", [False, True])
def test_all_composed_mhc_candidates_preserve_oracle_and_exact_replay(norm_dtype, has_low):
    from dataclasses import replace

    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.testing.mhc import post_reference, pre_reference

    from ignition.components.mhc import MHCScratch, MHCWeights
    from ignition.components.mhc_sm103 import (
        Query,
        _layout,
        _source_identity,
        _split_weights,
        _views,
        make_plan,
        tactics,
    )
    from ignition.components.native_operator import NativeConfig

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("composed mHC serves SM103")
    torch.manual_seed(8104)
    rows, hidden, device = 513, 4096, torch.device("cuda:0")
    fn = torch.randn(24, 4 * hidden, device=device) * .02
    if not has_low:
        fn = fn.bfloat16().float()
    weights = MHCWeights(fn, torch.rand(3, device=device) * .5,
                         torch.randn(24, device=device) * .3,
                         (torch.rand(hidden, device=device) + .5).to(norm_dtype))
    high, low, actual_has_low = _split_weights(fn)
    assert actual_has_low == has_low
    inputs = ((torch.randn(rows, hidden, device=device) * .1).bfloat16(),
              (torch.randn(rows, 4, hidden, device=device) * .1).bfloat16(),
              torch.rand(rows, 4, device=device) * 2,
              torch.softmax(torch.randn(rows, 4, 4, device=device), -1))
    saved_inputs = tuple(value.clone() for value in inputs)
    scratch = MHCScratch.allocate(rows, hidden, device)
    parameters = (high, low, weights.scale, weights.base, weights.norm)
    outputs = tuple(scratch.bind_args().values())
    q = Query(rows, hidden, "post_pre", 1e-6, 1e-6, 20, 1e-5,
              str(norm_dtype).removeprefix("torch."), has_low, _source_identity())
    legal = tactics(q)
    workspace = torch.empty(max(_layout(q, tactic)[1] for tactic in legal), dtype=torch.uint8, device=device)
    plans = [replace(make_plan(q, device), override=NativeConfig(tactic)) for tactic in legal]
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare(tuple(plan.request(name=f"candidate{index}", prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(inputs, parameters, outputs, workspace))) for index, plan in enumerate(plans)))
        session.freeze()
        mixed = post_reference(*inputs)
        collapsed, post, comb = pre_reference(mixed, fn, weights.scale, weights.base,
            rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
        wide = collapsed.float()
        normalized = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + 1e-5)
                      * weights.norm.float()).bfloat16()
        for plan in plans:
            state = plan.prepared.state
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                state.run(inputs, parameters, outputs, workspace)
            graph.replay()
            torch.cuda.synchronize()
            for got, want, bound in zip(outputs, (mixed, post, comb, normalized), (1e-3, 1e-3, 1e-3, 5e-3), strict=True):
                assert torch.isfinite(got).all(), state.tactic
                assert ((got.float() - want.float()).norm() / want.float().norm()).item() < bound, state.tactic
            checked = (*outputs, *_views(workspace, q, state.tactic)[:2])
            expected = tuple(value.clone() for value in checked)
            allocated = torch.cuda.memory_allocated()
            for _ in range(4):
                workspace.fill_(255)
                for output in outputs:
                    output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                for got, want in zip(checked, expected, strict=True):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            for got, want in zip(inputs, saved_inputs, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_compiled_composed_mhc_keeps_shared_scratch_and_layer_bindings():
    from b12x.preparation import PreparationSession

    from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
    from ignition.preparation.scratch import ScratchArena

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("composed mHC serves SM103")
    torch.manual_seed(8105)
    rows, hidden, device = 257, 4096, torch.device("cuda:0")
    residual = torch.randn(rows, 4, hidden, device=device).bfloat16()
    arena = ScratchArena(device)
    operators = []
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        for index in range(2):
            weights = MHCWeights((torch.randn(24, 4 * hidden, device=device) * .02).bfloat16().float(),
                torch.rand(3, device=device) * .5, torch.randn(24, device=device) * .3,
                (torch.rand(hidden, device=device) + .5).bfloat16())
            operators.append(PreparedMHC((residual,), weights, MHCScratch.allocate(rows, hidden, device),
                session=session, name=f"layer{index}", workspace=arena, rms_eps=1e-6, hc_eps=1e-6,
                sinkhorn_iters=20, norm_eps=1e-5))
        arena.reserve(arena.nbytes + 256)
        arena.finalize()
        arena.freeze()
        session.freeze()
        assert operators[0].composed.workspace.data_ptr() == operators[1].composed.workspace.data_ptr()
        assert operators[0].composed.parameters[0].data_ptr() != operators[1].composed.parameters[0].data_ptr()
        expected = []
        for operator in operators:
            operator(residual)
            expected.append(tuple(value.clone() for value in operator.scratch.bind_args().values()))

        def forward(value):
            return operators[0](value), operators[1](value)

        compiled = torch.compile(forward, fullgraph=True)
        compiled(residual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(residual)
        allocated = torch.cuda.memory_allocated()
        for _ in range(4):
            arena.tensor.fill_(255)
            for operator in operators:
                for output in operator.scratch.bind_args().values():
                    output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            for operator, wants in zip(operators, expected, strict=True):
                for got, want in zip(operator.scratch.bind_args().values(), wants, strict=True):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [513, 4096])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_two_token_finalizer_is_bitwise_identical_to_the_one_token_finalizer(rows, norm_dtype):
    from dataclasses import replace

    from b12x.preparation import PreparationSession, PreparedCall

    from ignition.components.mhc import MHCScratch, MHCWeights
    from ignition.components.mhc_sm103 import (
        Query,
        _layout,
        _source_identity,
        _split_weights,
        _views,
        make_plan,
        tactics,
    )
    from ignition.components.native_operator import NativeConfig

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("composed mHC serves SM103")
    torch.manual_seed(8106 + rows)
    hidden, device = 4096, torch.device("cuda:0")
    fn = (torch.randn(24, 4 * hidden, device=device) * .02).bfloat16().float()
    weights = MHCWeights(fn, torch.rand(3, device=device) * .5, torch.randn(24, device=device) * .3,
                         (torch.rand(hidden, device=device) + .5).to(norm_dtype))
    high, low, has_low = _split_weights(fn)
    inputs = ((torch.randn(rows, hidden, device=device) * .1).bfloat16(),
              (torch.randn(rows, 4, hidden, device=device) * .1).bfloat16(),
              torch.rand(rows, 4, device=device) * 2,
              torch.softmax(torch.randn(rows, 4, 4, device=device), -1))
    parameters = (high, low, weights.scale, weights.base, weights.norm)
    q = Query(rows, hidden, "post_pre", 1e-6, 1e-6, 20, 1e-5,
              str(norm_dtype).removeprefix("torch."), has_low, _source_identity())
    pairs = [(t, (*t[:5], "fused", 160)) for t in tactics(q) if t[5] == "fused_pair"]
    assert pairs
    workspace = torch.empty(max(_layout(q, t)[1] for pair in pairs for t in pair),
                            dtype=torch.uint8, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        for index, (two, one) in enumerate(pairs):
            results = []
            for tactic in (one, two):
                plan = replace(make_plan(q, device), override=NativeConfig(tactic))
                scratch = MHCScratch.allocate(rows, hidden, device)
                outputs = tuple(scratch.bind_args().values())
                session.prepare((plan.request(name=f"finalizer{index}.{tactic[5]}", prepare_call=lambda state, outputs=outputs: PreparedCall(
                    run=lambda: state.run(inputs, parameters, outputs, workspace))),))
                state = plan.prepared.state
                workspace.fill_(255)
                state.run(inputs, parameters, outputs, workspace)
                torch.cuda.synchronize()
                results.append((tuple(v.clone() for v in outputs),
                                tuple(v.clone() for v in _views(workspace, q, tactic)[:2])))
            (one_outputs, one_scratch), (two_outputs, two_scratch) = results
            for got, want in zip((*two_scratch, *two_outputs), (*one_scratch, *one_outputs), strict=True):
                assert torch.equal(got, want), pairs[index]
