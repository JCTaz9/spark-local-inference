"""Export the prepared mHC composition for exact replay after Python exits."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from b12x.preparation import PreparationSession
from b12x.testing.mhc import post_reference, pre_reference
from fixed_package_fixture import allocation_reference
from fixture_frontend import server_config
from signing import publisher

from ignition.components import mhc_sm103
from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
from ignition.components.native_operator import NativeConfig
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.package_file import combine
from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.runtime_package import export_model
from ignition.preparation.scratch import ScratchArena


def reference(inputs, weights):
    residual = post_reference(*inputs)
    collapsed, post, comb = pre_reference(residual, weights.fn, weights.scale, weights.base,
        rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20)
    wide = collapsed.float()
    normalized = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + 1e-5)
                  * weights.norm.float()).bfloat16()
    return residual, post, comb, normalized


def check_outputs(outputs, expected):
    for got, want, limit in zip(outputs, expected, (1e-3, 1e-3, 1e-3, 5e-3), strict=True):
        assert torch.isfinite(got).all()
        error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-6)
        assert error.item() < limit


@torch.no_grad()
def prepare(root, finalizer, precision):
    root.mkdir(parents=True, exist_ok=False)
    with ModuleImages() as images:
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (10, 3):
            raise RuntimeError("composed mHC package qualification requires SM103")
        device = torch.device("cuda", 0)
        with PackageStorage() as storage, PreparationSession(
                device=device, autotune=False, compile_workers=0) as session:
            torch.manual_seed(4917)
            rows, hidden = 320, 4096
            full_fp32 = precision == "fp32"
            fn = torch.randn(24, 4 * hidden, device=device) * .02
            if not full_fp32:
                fn = fn.bfloat16().float()
            norm = torch.rand(hidden, device=device) + .5
            if not full_fp32:
                norm = norm.bfloat16()
            else:
                assert torch.count_nonzero(norm != norm.bfloat16().float())
            weights = MHCWeights(fn, torch.rand(3, device=device) * .5,
                                 torch.randn(24, device=device) * .3, norm)
            inputs = ((torch.randn(rows, hidden, device=device) * .1).bfloat16(),
                      (torch.randn(rows, 4, hidden, device=device) * .1).bfloat16(),
                      torch.rand(rows, 4, device=device) * 2,
                      torch.softmax(torch.randn(rows, 4, 4, device=device), -1))
            outputs = MHCScratch.allocate(rows, hidden, device)
            arena = ScratchArena(device, overlap=False)
            tactic = (64, 128, 4, 4, 1 if full_fp32 else 4,
                      finalizer, 160 if finalizer == "fused" else 128)
            make_plan = mhc_sm103.make_plan

            def pinned_plan(query, target_device):
                # Only selection is pinned: declaration, compilation, materialization,
                # final binding and the captured custom op remain the production path.
                return replace(make_plan(query, target_device), override=NativeConfig(tactic))

            with patch.object(mhc_sm103, "make_plan", pinned_plan):
                operation = PreparedMHC(inputs, weights, outputs, session=session,
                    name="mhc", workspace=arena, rms_eps=1e-5, hc_eps=1e-6,
                    sinkhorn_iters=20, norm_eps=1e-5)
            arena.finalize()
            arena.freeze()
            composed = operation.composed
            assert composed is not None and composed.has_low == full_fp32
            assert composed.state.tactic == tactic
            workspace = composed.workspace
            targets = tuple(outputs.bind_args().values())
            expected_values = reference(inputs, weights)
            verification = torch.empty(sum(value.nbytes for value in targets),
                                       dtype=torch.uint8, device=device)

            def run():
                workspace.fill_(0xA5)
                for value in targets:
                    value.fill_(float("nan"))
                operation(*inputs)
                offset = 0
                for value in targets:
                    verification[offset:offset + value.nbytes].copy_(value.view(torch.uint8).flatten())
                    offset += value.nbytes

            run()
            torch.cuda.synchronize()
            check_outputs(targets, expected_values)
            session.freeze()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            try:
                with session.capture(), torch.cuda.graph(graph, pool=storage.graph_pool.id):
                    run()
                graph.instantiate()
                graph.replay()
                torch.cuda.synchronize()
                check_outputs(targets, expected_values)
                expected = verification.cpu().numpy().tobytes()
                for _ in range(3):
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key], key
                    assert verification.cpu().numpy().tobytes() == expected
                buffers = {
                    "frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                    "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                    "completion_records": torch.zeros((2, 1, 24), dtype=torch.int64, device=device),
                    "output": verification,
                }
                prepared = PreparedExecution(
                    device=0, graphs={"prefill": graph}, buffers=buffers,
                    signatures={"prefill": GraphSignature("prefill", 1, rows)}, max_active=1,
                    resources=(session, operation, weights, inputs, outputs, arena, workspace, verification),
                    export_zero_ranges=((workspace.data_ptr(), workspace.nbytes),
                                        (verification.data_ptr(), verification.nbytes)),
                )
                prepared.server_config = server_config(root)
                prepared.package_data = None
                package = export_model(prepared, images, storage, root / "rank.json", options={}, sources=True)
                assert package.export_counts["modules"] > 0
                combine(root / "package", [root / "rank.json"], [], publisher=publisher())
                (root / "expected.json").write_text(json.dumps({
                    "reference": allocation_reference(package.allocations, verification),
                    "bytes": list(expected),
                    "addresses": {allocation.name: allocation.address for allocation in package.allocations},
                    "zero_reference": allocation_reference(package.allocations, workspace),
                    "zero_bytes": workspace.nbytes,
                }))
                print(f"exported mHC {finalizer} {precision}: all four outputs", flush=True)
            finally:
                graph.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("finalizer", choices=("fused", "separate"))
    parser.add_argument("precision", choices=("bf16", "fp32"))
    args = parser.parse_args()
    prepare(args.directory, args.finalizer, args.precision)
