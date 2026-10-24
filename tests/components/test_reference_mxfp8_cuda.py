"""Reference FP8 attention preparation, compilation and native graph lifetimes."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)


@torch.library.custom_op(
    "flare_test::reference_parallel_poison", mutates_args=("target",)
)
def _poison_parallel_bytes(target: torch.Tensor) -> None:
    target.fill_(255)


@_poison_parallel_bytes.register_fake
def _poison_parallel_bytes_fake(target):
    pass


def _reference_quantize(source):
    """Checkpoint inference/kernel.py::act_quant, E4M3/g32/UE8M0 mode."""
    import torch

    x = source.float().reshape(source.shape[0], -1, 32)
    maximum = x.abs().amax(-1).clamp_min(1e-4)
    bits = (
        maximum * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=source.device)
    ).view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).int()
    scale = ((exponent + 127) << 23).view(torch.float32)
    return (
        (x / scale[:, :, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape_as(source),
        (exponent + 127).to(torch.uint8),
    )


@torch.no_grad()
def test_reference_projection_capacities_and_native_lifetimes(tmp_path):
    import torch

    native = pytest.importorskip("flare_heliopause")
    if not hasattr(native, "GraphReplayWorker"):
        pytest.skip("requires the qualification extension")
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.preparation.scratch import ScratchArena, GraphBufferPool
    from ignition.preparation.aot import aot_forward
    from ignition.components.reference_mxfp8 import (
        ReferenceMXFP8Projection,
        ReferenceQueryNormQuant,
        ReferenceProjectionChain,
    )
    from ignition.components.rms_norm import PreparedRMSNorm
    from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

    reference_torch = _reference_quantize
    SNAPSHOT = Path(
        os.environ.get(
            "FLARE_DEEPSEEK_TEST_SNAPSHOT",
            str(
                Path.home()
                / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
            ),
        )
    )
    if not SNAPSHOT.is_dir():
        pytest.skip("DeepSeek V4.1 source checkpoint is unavailable")

    def check(actual, expected):
        d = (actual.float() - expected.float()).abs()
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
        return dict(
            max_abs=d.max().item(),
            relative_rms=(
                d.square().mean().sqrt() / expected.float().square().mean().sqrt()
            ).item(),
        )

    torch.manual_seed(692713)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    with DeepSeekWeightLoader(
        inspect_indexed_checkpoint(SNAPSHOT),
        torch.device("cuda:0"),
        shard=TensorParallelShard(0, 4),
        prefixes=("layers.0.attn.",),
    ) as loader:
        a = loader.load_linear_group(("layers.0.attn.wq_a", "layers.0.attn.wkv"))
        b = loader.load_linear_group(
            ("layers.0.attn.wq_b",), row_ranges=((0, 8192),), pad=True
        )
        qweight = loader.load_tensor("layers.0.attn.q_norm.weight")
        kweight = loader.load_tensor("layers.0.attn.kv_norm.weight")
    session = RecordedPreparationSession(
        device="cuda:0",
        checkpoint=str(SNAPSHOT),
        workload_schema="reference-components-" + tmp_path.name,
        numerical_policy="hf-fp8-floor-bf16-boundary",
        autotune_effort="medium",
        compile_workers=4,
    )
    arena = ScratchArena(torch.device("cuda:0"))
    session.register_scratch(arena)
    pool = GraphBufferPool(torch.device("cuda:0"))
    cases = []
    for rows in (1, 2, 3, 4, 5, 6, 7, 8, 16, 129):
        cursor = pool.cursor()
        source = cursor.zeros((rows, 5120), torch.bfloat16)
        projected = cursor.zeros((rows, 1792), torch.bfloat16)
        query = cursor.zeros((rows, 1280), torch.bfloat16)
        kv = cursor.zeros((rows, 512), torch.bfloat16)
        out = cursor.zeros((rows, 8192), torch.bfloat16)
        norm = ReferenceQueryNormQuant(
            projected, (qweight, kweight), query, kv, buffers=cursor
        )
        first = ReferenceMXFP8Projection(
            source, a, projected, session=session, name=f"qkv-{rows}", workspace=arena
        )
        last = ReferenceMXFP8Projection(
            query,
            b,
            out,
            session=session,
            name=f"qb-{rows}",
            workspace=arena,
            quantized=(norm.quantized, norm.scales),
        )
        poison = cursor.zeros((1024,), torch.uint8)
        main_out = cursor.zeros((rows, 512), torch.bfloat16)
        main = PreparedRMSNorm(
            kv, kweight, main_out, eps=1e-20, session=session, name=f"parallel-{rows}"
        )
        side = ReferenceProjectionChain(
            last, parallel_plans=(main.plan,), session=session, workspace=arena
        )
        cases.append(
            (
                rows,
                source,
                projected,
                query,
                kv,
                out,
                norm,
                first,
                last,
                main,
                side,
                poison,
            )
        )
    assert len({case[6].backing.untyped_storage()._cdata for case in cases}) == 1
    from b12x.gemm._tuning import _codegen_snapshot

    global_codegen = _codegen_snapshot()
    assert all(
        first.plan.query.codegen == global_codegen
        and last.plan.query.codegen == global_codegen
        for rows, source, projected, query, kv, out, norm, first, last, main, side, poison in cases
    )
    session.tune(show_progress=False)
    assert _codegen_snapshot() == global_codegen
    for (
        rows,
        source,
        projected,
        query,
        kv,
        out,
        norm,
        first,
        last,
        main,
        side,
        poison,
    ) in cases:
        if rows <= 8:
            assert (
                first.plan.selection.source == last.plan.selection.source == "override"
            )
        else:
            assert first.plan.prepared.state.dense.lowering.policy.split_k_slices == 1
            assert last.plan.prepared.state.dense.lowering.policy.split_k_slices == 1
    report = {"passed": False, "cases": [], "tuning_report": session.tuning_report}
    retained = []
    all_graphs = {}
    for c in cases:
        (
            rows,
            source,
            projected,
            query,
            kv,
            out,
            norm,
            first,
            last,
            main,
            side,
            poison,
        ) = c
        norm.backing.fill_(255)
        source.copy_(torch.randn_like(source) * 0.125)
        first(source)
        qv, qs = reference_torch(source)
        xd = (
            qv.float().reshape(rows, 160, 32)
            * qs.view(torch.float8_e8m0fnu).float()[:, :, None]
        ).reshape(rows, 5120)
        aw = a.weight
        ad = (
            aw.values.float().reshape(1792, 160, 32)
            * aw.scale_rows.view(torch.float8_e8m0fnu).float().reshape(1792, 160, 1)
        ).reshape(1792, 5120)
        first_error = check(projected, (xd @ ad.T).to(torch.bfloat16))
        serial = Forward(
            first, last, norm, main, source, projected, query, kv, poison, side=None
        )
        parallel = Forward(
            first, last, norm, main, source, projected, query, kv, poison, side=side
        )
        for f in (serial, parallel):
            f()
        torch.cuda.synchronize()
        # Validate the actual materialized normalized input's exact FP8 encoding.
        expected_q, expected_s = reference_torch(query)
        torch.testing.assert_close(
            norm.quantized.view(torch.uint8),
            expected_q.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(norm.scale_rows, expected_s, rtol=0, atol=0)
        x = (
            norm.quantized.float().reshape(rows, 40, 32)
            * norm.scale_rows.view(torch.float8_e8m0fnu).float()[:, :, None]
        ).reshape(rows, 1280)
        bw = b.weight
        wd = (
            bw.values.float().reshape(8192, 40, 32)
            * bw.scale_rows.view(torch.float8_e8m0fnu).float().reshape(8192, 40, 1)
        ).reshape(8192, 1280)
        oracle = (x @ wd.T).to(torch.bfloat16)
        error = check(out, oracle)
        compiled = []
        for f in (serial, parallel):
            compiled.append(aot_forward(f))
            compiled[-1](f)
        torch.cuda.synchronize()
        graphs = []
        for f, compiled_forward in zip((serial, parallel), compiled):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with session.capture(), torch.cuda.graph(graph):
                compiled_forward(f)
            graph.instantiate()
            graphs.append(graph)
        # Change inputs, poison transient outputs, and alternate compiled paths.
        for scale in (0.0, 0.125, 1e-20, 1.0):
            norm.backing.fill_(255)
            source.copy_(torch.randn_like(source) * scale)
            graphs[0].replay()
            torch.cuda.synchronize()
            ref_out = out.clone()
            ref_kv = kv.clone()
            for _ in range(4):
                projected.fill_(float("nan"))
                query.fill_(float("nan"))
                out.fill_(float("nan"))
                kv.fill_(float("nan"))
                norm.backing.fill_(255)
                graphs[1].replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
                torch.testing.assert_close(kv, ref_kv, rtol=0, atol=0)
                expected_q, expected_s = reference_torch(query)
                torch.testing.assert_close(
                    norm.quantized.view(torch.uint8),
                    expected_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(norm.scale_rows, expected_s, rtol=0, atol=0)
            del ref_out, ref_kv
        report["cases"].append(
            dict(
                rows=rows,
                source_gemv=rows <= 8,
                reference_dense_fallback=rows > 8,
                compiled_serial_parallel_exact=True,
                quantization_exact=True,
                qkv_oracle_error=first_error,
                oracle_error=error,
            )
        )

        print("PASS_ROWS", rows, flush=True)
        retained.append((c, serial, parallel, compiled, graphs))
        all_graphs[f"{rows}-serial"] = graphs[0]
        all_graphs[f"{rows}-parallel"] = graphs[1]
    from ignition.preparation.graphs import PreparedGraphBundle

    # Register contiguous storage owners, including noncontiguous scale views.
    owners = [arena.tensor, qweight, kweight]
    owners.extend(chunk for chunks in pool.chunks.values() for chunk in chunks)
    owners.extend(
        (a.weight.values, a.weight.scale_mma, b.weight.values, b.weight.scale_mma)
    )
    for c, serial, parallel, compiled, graphs in retained:
        owners.extend((c[6].backing, c[7].alpha, c[8].alpha))
    buffers = {}
    seen = set()
    for t in owners:
        st = t.untyped_storage()
        if st.data_ptr() in seen:
            continue
        seen.add(st.data_ptr())
        flat = torch.empty(0, dtype=torch.uint8, device=t.device).set_(
            st, 0, (st.nbytes(),), (1,)
        )
        buffers[str(len(buffers))] = flat
    bundle = PreparedGraphBundle(
        device=0,
        graphs=all_graphs,
        buffers=buffers,
        resources=(retained, session, arena, pool, a, b, qweight, kweight),
    )
    worker = bundle.start_worker()
    try:
        names = list(all_graphs)
        native_order = (129, 1, 16, 7, 8, 2, 6, 3, 5, 4, 1, 129, 8, 7)
        by_rows = {entry[0][0]: index for index, entry in enumerate(retained)}
        for index in (by_rows[rows] for rows in native_order):
            c, serial, parallel, compiled, graphs = retained[index]
            rows, source, projected, query, kv, out = c[:6]
            c[6].backing.fill_(255)
            source.copy_(torch.randn_like(source) * 0.125)
            graphs[0].replay()
            torch.cuda.synchronize()
            expected_out = out.clone()
            expected_kv = kv.clone()
            projected.fill_(float("nan"))
            query.fill_(float("nan"))
            kv.fill_(float("nan"))
            out.fill_(float("nan"))
            c[6].backing.fill_(255)
            before = torch.cuda.memory_allocated()
            sequence = worker.submit(names.index(f"{rows}-parallel"), 1)
            completion = worker.poll(timeout_ms=30000)
            assert (
                completion is not None
                and completion.ticket == sequence
                and completion.error is None
            )
            torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
            torch.testing.assert_close(kv, expected_kv, rtol=0, atol=0)
            assert before == torch.cuda.memory_allocated()
            del expected_out, expected_kv
        report["native_alternating_capacities"] = [rows for rows in native_order]
        report["native_allocation_stable"] = True
    finally:
        worker.close()
    for graph in all_graphs.values():
        graph.reset()
    report["passed"] = True
    assert report["passed"] and report["native_allocation_stable"]
    torch.cuda.synchronize()
    session.close()


class Forward:
    def __init__(
        self, first, last, norm, main, source, projected, query, kv, poison, side
    ):
        self.first, self.last, self.norm, self.main = first, last, norm, main
        self.poison = poison
        self.source, self.projected, self.query, self.kv, self.side = (
            source,
            projected,
            query,
            kv,
            side,
        )

    def __call__(self):
        self.first(self.source)
        self.norm(self.projected)
        if self.side is None:
            self.last(self.query)
            self.main(self.kv)
            _poison_parallel_bytes(self.poison)
        else:
            self.side.start(self.query)
            self.main(self.kv)
            _poison_parallel_bytes(self.poison)
            self.side.finish(self.query, self.poison)
