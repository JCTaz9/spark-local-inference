"""Measure GLM's complete composed mHC and its captured stages on SM103.

Loads only one layer's small checkpoint mHC weights. Every sample receives a
fresh branch and coefficients produced by an actual preceding mHC pre graph.
Stage predecessors execute outside timing. Full-operation checks preserve the
BF16 residual/collapse boundaries and FP32 projection and mixing reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation._measurement import _l2_flush_fn
from b12x.testing.benchmark import measure_calls
from b12x.testing.mhc import post_reference, pre_reference
from safetensors import safe_open

from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
from ignition.components.mhc_sm103 import _norm_arguments, _project_options


def checkpoint_weights(checkpoint, layer, branch, device, *, full_fp32=False):
    names = (f"hc_{branch}_fn", f"hc_{branch}_scale", f"hc_{branch}_base",
             "input_layernorm.weight" if branch == "attn" else "post_attention_layernorm.weight")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    values = []
    for name in names:
        key = f"model.language_model.layers.{layer}.{name}"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as source:
            values.append(source.get_tensor(key))
    fn, scale, base, norm = values
    fn = fn.float()
    if full_fp32:
        generator = torch.Generator().manual_seed(8103)
        fn = fn + torch.randn(fn.shape, generator=generator) * 1e-4
    return MHCWeights(*(value.to(device) for value in (fn, scale.float(), base.float(), norm)))


def reference(inputs, weights, eps):
    mixed = inputs[0] if len(inputs) == 1 else post_reference(*inputs)
    collapsed, post, comb = pre_reference(mixed, weights.fn, weights.scale, weights.base,
        rms_eps=eps["rms_eps"], hc_eps=eps["hc_eps"], sinkhorn_iters=eps["sinkhorn_iters"])
    wide = collapsed.float()
    normalized = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps["norm_eps"])
                  * weights.norm.float()).bfloat16()
    return mixed, post, comb, normalized


def compare(actual, expected):
    result = {}
    limits = (1e-3, 1e-3, 1e-3, 5e-3)
    for name, got, want, limit in zip(("residual", "post", "comb", "normalized"),
                                    actual, expected, limits, strict=True):
        delta = got.float() - want.float()
        relative = (delta.norm() / want.float().norm()).item()
        result[name] = {"relative_l2": relative, "max_abs": delta.abs().max().item(),
                        "reference_norm": want.float().norm().item(), "relative_l2_limit": limit}
        if not torch.isfinite(got).all() or not relative < limit:
            raise AssertionError(f"{name}: {result[name]}")
    return result


def benchmark(args, rows, branch, *, weight_owners):
    device = torch.device("cuda", 0)
    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    eps = {"rms_eps": config["rms_norm_eps"], "hc_eps": config["hc_eps"],
           "sinkhorn_iters": config["hc_sinkhorn_iters"], "norm_eps": config["rms_norm_eps"]}
    weights = checkpoint_weights(args.checkpoint, args.layer, branch, device,
                                 full_fp32=args.weight_mode == "full-fp32")
    producer_weights = checkpoint_weights(args.checkpoint, args.layer,
        "ffn" if branch == "attn" else "attn", device)
    # Retain original checkpoint tensors across cases, matching prepared model ownership.
    weight_owners.extend((weights, producer_weights))
    generator = torch.Generator(device=device).manual_seed(3187)
    seed_residual = (torch.randn(rows, 4, 4096, device=device, generator=generator) * .1).bfloat16()
    seed_branch = (torch.randn(rows, 4096, device=device, generator=generator) * .1).bfloat16()
    residual = torch.empty_like(seed_residual)
    source = torch.empty_like(seed_branch)
    previous = MHCScratch.allocate(rows, 4096, device)
    scratch = MHCScratch.allocate(rows, 4096, device)
    inputs = (source, previous.residual, previous.post, previous.comb)
    with PreparationSession(device=device, autotune=args.autotune, compile_workers=0) as session:
        producer = PreparedMHC((residual,), producer_weights, previous,
            session=session, name="producer", workspace=None, **eps)
        operation = PreparedMHC(inputs, weights, scratch,
            session=session, name="post_pre", workspace=None, **eps)
        session.freeze()
        if producer.composed is None or operation.composed is None:
            raise RuntimeError("benchmark requires the composed SM103 prefill path")
        composed = operation.composed

        def produce():
            residual.copy_(seed_residual)
            source.copy_(seed_branch)
            producer(residual)

        produce()
        operation(*inputs)
        torch.cuda.synchronize()

        def capture(run):
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                run()
            return graph

        producer_graph = capture(produce)
        from flashinfer.mhc import (
            _mhc_post_impl,
            _mhc_pre_big_fuse_impl,
            _mhc_pre_big_fuse_with_norm_impl,
        )

        state, q, tactic = composed.state, composed.query, composed.state.tactic
        projection_args = (scratch.residual, composed.high, composed.low, composed.dot, composed.squares, rows)
        projection_grid = ((rows + tactic[0] - 1) // tactic[0], tactic[2], 1)
        stages = [
            ("mix", lambda: _mhc_post_impl(scratch.residual, *inputs)),
            ("project", lambda: state.projection[projection_grid](*projection_args)),
        ]
        if tactic[5] in ("fused", "fused_pair"):
            stages.append(("finalize_norm", lambda: _mhc_pre_big_fuse_with_norm_impl(
                scratch.post, scratch.comb, scratch.normalized, composed.dot, composed.squares,
                scratch.residual, weights.scale, weights.base, weights.norm,
                q.rms_eps, q.hc_eps, q.hc_eps, 2.0, q.sinkhorn_iters, q.norm_eps, tactic[2], tactic[6],
                2 if tactic[5] == "fused_pair" else 1)))
        else:
            stages.append(("big_fuse", lambda: _mhc_pre_big_fuse_impl(
                scratch.post, scratch.comb, composed.collapsed, composed.dot, composed.squares,
                scratch.residual, weights.scale, weights.base, 4 * q.hidden,
                q.rms_eps, q.hc_eps, q.hc_eps, 2.0, q.sinkhorn_iters, tactic[2], tactic[6])))
            if weights.norm.dtype == torch.float32:
                norm_args, _ = _norm_arguments(q, composed.collapsed, weights.norm, scratch.normalized)
                stages.append(("norm", lambda: state.norm[(rows, 1, 1)](*norm_args)))
            else:
                from ignition.components.mhc_sm103_norm import launch

                stages.append(("norm", lambda: launch(state.norm, composed.collapsed, weights.norm,
                    scratch.normalized, rows, q.norm_eps)))
        graphs = [capture(run) for _, run in stages]
        whole = capture(lambda: operation(*inputs))

        torch.backends.cuda.matmul.allow_tf32 = False
        expected_previous = reference((seed_residual,), producer_weights, eps)
        expected = reference((seed_branch, *expected_previous[:3]), weights, eps)
        producer_graph.replay()
        whole.replay()
        torch.cuda.synchronize()
        outputs = tuple(scratch.bind_args().values())
        correctness = compare(outputs, expected)
        exact = tuple(value.clone() for value in outputs)
        allocated = torch.cuda.memory_allocated(device)
        for _ in range(4):
            producer_graph.replay()
            composed.workspace.fill_(255)
            for value in outputs:
                value.fill_(float("nan"))
            whole.replay()
            torch.cuda.synchronize()
            if torch.cuda.memory_allocated(device) != allocated:
                raise AssertionError("captured mHC allocated storage")
            for got, want in zip(outputs, exact, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

        calls = {}
        for index, ((name, _), graph) in enumerate(zip(stages, graphs, strict=True)):
            def stage_produce(index=index):
                producer_graph.replay()
                for prior in graphs[:index]:
                    prior.replay()
            calls[name] = PreparedCall(run=graph.replay, produce=stage_produce,
                                      owners=(graph, operation, producer_graph))
        calls["whole"] = PreparedCall(run=whole.replay, produce=producer_graph.replay,
                                      output=outputs, owners=(whole, operation, producer_graph))
        measurements = {}
        for cache in ("warm", "cold"):
            result = measure_calls(calls, device=device, samples=args.samples, rounds=args.rounds,
                                   eviction=_l2_flush_fn(device, enabled=cache == "cold"))
            measurements[cache] = {name: {"median_us": result.latencies_us[name],
                                         "samples_us": result.raw_samples(name)} for name in calls}
        kernels = []
        if args.profile:
            producer_graph.replay()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                whole.replay()
                torch.cuda.synchronize()
            trace = args.output.with_name(f"{args.output.stem}-{rows}-{branch}.trace.json")
            profiler.export_chrome_trace(str(trace))
            kernels = [{"name": event.name, "duration_us": event.device_time_total}
                       for event in profiler.events()
                       if event.device_type == torch.autograd.DeviceType.CUDA]
        producer_graph.replay()
        whole.replay()
        torch.cuda.synchronize()
        compare(outputs, expected)
        return {"rows": rows, "hidden": 4096, "streams": 4, "operation": "post_pre",
                "checkpoint": str(args.checkpoint), "layer": args.layer, "branch": branch,
                "norm_dtype": str(weights.norm.dtype), "weight_mode": args.weight_mode,
                "fn_low_nonzero": int(torch.count_nonzero(composed.low).item()),
                "projection_config": _project_options(q, tactic), "selected_tactic": tactic,
                "autotune": args.autotune,
                "eps": eps, "correctness": correctness, "measurements": measurements,
                "profile_kernels": kernels,
                "scope": "Captured complete PreparedMHC and its stages; real preceding pre graph and fresh branch copies outside timing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/checkpoints/GLM-5.3-Flash-FP4-Local"))
    parser.add_argument("--rows", type=int, action="append")
    parser.add_argument("--branch", choices=("attn", "ffn"), action="append")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--weight-mode", choices=("checkpoint", "full-fp32"), default="checkpoint")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--autotune", action="store_true", help="race complete mHC compositions before measurement")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability(0) != (10, 3):
        parser.error("requires assigned SM103 GPU")
    if args.output.exists():
        parser.error("output must be fresh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    weight_owners = []
    for rows in args.rows or (4096, 8192):
        for branch in args.branch or ("attn", "ffn"):
            result = benchmark(args, rows, branch, weight_owners=weight_owners)
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({key: result[key] for key in ("rows", "branch", "norm_dtype", "fn_low_nonzero")}
                | {cache: {name: v["median_us"] for name, v in values.items()}
                   for cache, values in result["measurements"].items()}), flush=True)


if __name__ == "__main__":
    main()
