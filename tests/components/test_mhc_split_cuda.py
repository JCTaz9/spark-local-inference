"""Split lagged normalization and coefficient execution preserve complete outputs."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession
from b12x.testing.mhc import make_inputs

from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
from ignition.components.mhc_overlap import ConcurrentMHC
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.decoder import Scratch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 7, 8, 16, 32])
@pytest.mark.parametrize("kind", ["pre", "post_pre"])
@torch.no_grad()
def test_split_lagged_complete_output_parity(rows, kind):
    device = torch.device("cuda:0")
    residual, branch, fn, scale, base = make_inputs(
        tokens=rows, hidden_size=5120, seed=41063, device=device)
    norm = torch.linspace(.5, 1.5, 5120, device=device).bfloat16()
    post = torch.ones((rows, 4), device=device)
    comb = torch.eye(4, device=device).expand(rows, -1, -1).contiguous()
    incoming = torch.rand((rows, 4), device=device)
    inputs = (residual,) if kind == "pre" else (branch, residual, post, comb)
    buffers = [MHCScratch.allocate(rows, 5120, device) for _ in range(2)]
    predicted = [torch.empty_like(incoming) for _ in range(2)]
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operations = [PreparedMHC(inputs, MHCWeights(fn, scale, base, norm), output,
            session=session, name=f"{kind}.{split}", workspace=None,
            rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-6,
            pre_mix=incoming, pre_out=target, split_lagged=split)
            for split, output, target in zip((False, True), buffers, predicted, strict=True)]

        def split_forward():
            operations[1].input(*inputs)
            operations[1].coefficients()

        forwards = [torch.compile(lambda: operations[0](*inputs), fullgraph=True),
                    torch.compile(split_forward, fullgraph=True)]
        graphs = []
        for forward in forwards:
            forward()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                forward()
            graphs.append(graph)
        try:
            for step in range(4):
                residual.normal_(std=.1 * (step + 1))
                branch.normal_(std=.2)
                post.uniform_(.2, 1.8)
                comb.uniform_(.05, .7)
                incoming.uniform_(.1, .9)
                fn.add_(.000001)
                norm.mul_(1.001)
                scale.mul_(1.01)
                base.add_(.002)
                for output, target in zip(buffers, predicted, strict=True):
                    for value in (*output.bind_args().values(), target):
                        value.fill_(float("nan"))
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                for expected, actual in zip(
                        (*buffers[0].bind_args().values(), predicted[0]),
                        (*buffers[1].bind_args().values(), predicted[1]), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            for graph in graphs:
                graph.reset()


@pytest.mark.parametrize("rows", [1, 7, 8, 16, 32])
@torch.no_grad()
def test_concurrent_coefficients_native_private_scratch_lifetime(rows):
    device = torch.device("cuda:0")
    residual, branch, fn, scale, base = make_inputs(
        tokens=rows, hidden_size=5120, seed=41064, device=device)
    norm = torch.linspace(.5, 1.5, 5120, device=device).bfloat16()
    post = torch.ones((rows, 4), device=device)
    comb = torch.eye(4, device=device).expand(rows, -1, -1).contiguous()
    incoming = torch.rand((rows, 4), device=device)
    inputs = branch, residual, post, comb
    outputs = [MHCScratch.allocate(rows, 5120, device) for _ in range(2)]
    predicted = [torch.empty_like(incoming) for _ in range(2)]
    consumer = [torch.empty_like(branch) for _ in range(2)]
    private = torch.empty((rows * 80 * 25 * 4,), device=device, dtype=torch.uint8)
    workspace = torch.empty((1 << 20,), device=device, dtype=torch.uint8)
    stream_owner = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operations = [PreparedMHC(inputs, MHCWeights(fn, scale, base, norm), output,
            session=session, name=f"native.{split}", workspace=private if split else None,
            rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-6,
            pre_mix=incoming, pre_out=target, split_lagged=split)
            for split, output, target in zip((False, True), outputs, predicted, strict=True)]
        concurrent = ConcurrentMHC(operations[1], stream=stream_owner.auxiliary_stream)

        def control():
            value = operations[0](*inputs)
            torch.mul(value, value, out=consumer[0])

        def candidate():
            value = concurrent.start(*inputs)
            torch.mul(value, value, out=consumer[1])
            # Main operator storage can be overwritten while coefficients run.
            workspace.fill_(91)
            for value in (*inputs, incoming):
                value.fill_(float("nan"))
            concurrent.finish(consumer[1])
            # The join is the last reader of private projection partials.
            private.fill_(165)

        compiled = [compile_forward(control), compile_forward(candidate)]
        graphs = {}
        for name, forward in zip(("control", "candidate"), compiled, strict=True):
            forward()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                forward()
            graphs[name] = graph
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={},
            resources=(operations, concurrent, stream_owner, private, workspace, compiled))
        worker = bundle.start_worker()
        try:
            for step in range(16):
                residual.normal_(std=.1 * (1 + step % 5))
                branch.normal_(std=.2)
                post.uniform_(.2, 1.8)
                comb.uniform_(.05, .7)
                incoming.uniform_(.1, .9)
                private.fill_(90)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for name in ("control", "candidate"):
                    ticket = worker.submit(bundle.graph_indices[name])
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for expected, actual in zip(
                        (*outputs[0].bind_args().values(), predicted[0], consumer[0]),
                        (*outputs[1].bind_args().values(), predicted[1], consumer[1]), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.all(private == 165)
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()


@torch.no_grad()
def test_alternating_recipe_scratch_reset_and_carry_boundaries_native():
    """Exercise both alternating mHC buffers with the decoder's pooled layout."""
    device = torch.device("cuda:0")
    source, _, fn, scale, base = make_inputs(tokens=32, hidden_size=5120, seed=41065, device=device)
    weights = MHCWeights(fn, scale, base, torch.ones(5120, device=device, dtype=torch.bfloat16))
    storage = [Scratch.allocate(32, 1, 4, device) for _ in range(2)]
    arena = ScratchArena(device)
    assert arena.background_stream is arena.background_stream
    assert arena.background_stream.cuda_stream != arena.auxiliary_stream.cuda_stream
    assert arena.background_stream.cuda_stream != torch.cuda.current_stream(device).cuda_stream
    graphs, owners = {}, []
    with RecordedPreparationSession(device=device, checkpoint="mhc-boundary-state",
            workload_schema="mhc.reset-carry", numerical_policy="fp32-fn",
            autotune_effort="off", compile_workers=2) as session:
        session.register_scratch(arena)
        forwards = {}
        for rows in (1, 8, 32):
            for arm, original in enumerate(storage):
                s = original.rows(rows, 1)
                calls = []
                for stage in range(4):
                    attention = stage % 2 == 0
                    incoming = ((s.residual,) if stage == 0 else
                        (s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if attention else
                        (s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb))
                    operation = PreparedMHC(incoming, weights, s.attention_hc if attention else s.ffn_hc,
                        session=session, name=f"rows{rows}.arm{arm}.stage{stage}",
                        workspace=s.mhc_partials.view(-1) if arm else arena,
                        rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-6,
                        pre_mix=s.pre_mix if attention else s.attn_mix,
                        pre_out=s.attn_mix if attention else s.pre_mix, split_lagged=bool(arm))
                    call = ConcurrentMHC(operation, stream=arena.background_stream) if arm else operation
                    calls.append((call, incoming, s.attention_output if attention else s.mlp_output))

                def forward(s=s, calls=calls, arm=arm, rows=rows):
                    s.residual.copy_(source[:rows])
                    s.pre_mix.fill_(.25)
                    for call, incoming, output in calls:
                        value = call.start(*incoming) if arm else call(*incoming)
                        torch.mul(value, .5, out=output)
                        arena.tensor.fill_(90)
                        if arm:
                            call.finish(output)
                            s.mhc_partials.fill_(165)

                forwards[f"{rows}.{arm}"] = forward
                owners.append((s, calls))
        session.tune(show_progress=False)
        for name, forward in forwards.items():
            compiled = compile_forward(forward)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[name] = graph
            owners.append(compiled)
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={},
            resources=(storage, owners, arena, weights, source))
        worker = bundle.start_worker()
        try:
            for rows in (32, 1, 8, 32, 8, 1):
                source.normal_(std=.2)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for arm in (0, 1):
                    ticket = worker.submit(bundle.graph_indices[f"{rows}.{arm}"])
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for name in ("pre_mix", "attn_mix", "attention_output", "mlp_output"):
                    torch.testing.assert_close(getattr(storage[1], name)[:rows],
                                               getattr(storage[0], name)[:rows], rtol=0, atol=0)
                for kind in ("attention_hc", "ffn_hc"):
                    for name in ("residual", "normalized", "post", "comb"):
                        torch.testing.assert_close(getattr(getattr(storage[1], kind), name)[:rows],
                                                   getattr(getattr(storage[0], kind), name)[:rows], rtol=0, atol=0)
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
