"""Compare a persistent decoder tail with separated prepared checkpoint operations."""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist


def main(checkpoint, rows, layer):
    from b12x.comm.pcie._cuda_ipc import CudaRTLibrary
    from flashinfer.experimental.persistent_decoder import PersistentDecoderTP4
    from ignition.components.attention_output import PreparedGroupedOutputProjection
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.linear import UnquantizedProjection, QuantizedProjection
    from ignition.components.mla_norm_quant import PreparedMLANormQuant
    from ignition.components.persistent_decoder import PersistentPrologueWeights
    from ignition.components.rotary import CompressedRotary
    from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
    from ignition.components.mlp import MLPScratch, PreparedMLP
    from ignition.components.moe import (
        RouterScratch,
        PreparedRouter,
        PreparedRoutedExperts,
    )
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.persistent_decoder import PersistentDecoderWeights
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = channel = None
    try:
        print("rank", rank, "loading", layer, flush=True)
        manifest = inspect_indexed_checkpoint(checkpoint)
        with DeepSeekWeightLoader(
            manifest,
            device,
            shard=TensorParallelShard(rank, 4),
            prefixes=(f"layers.{layer}.",),
        ) as loader:
            weights = loader.load_layer(layer)
            text = loader.config.text
            config = loader.config
        print("rank", rank, "loaded", flush=True)
        packed = PersistentDecoderWeights.pack(weights)
        print(
            "rank",
            rank,
            "dense_nbytes",
            packed.dense_nbytes,
            "rep",
            packed.expert_owner.w13_layout,
            [
                (tuple(t.shape), str(t.dtype), t.stride(0))
                for t in packed.tensors[11:15]
            ],
            flush=True,
        )
        channel = PersistentDecoderTP4(
            dist.group.WORLD, device=device, ipc=CudaRTLibrary()
        )
        runtime = create_all_reduce_runtime(
            dist.group.WORLD, device, max_tokens=rows, hidden_size=5120
        )
        arena = ScratchArena(device, overlap=False)
        generator = torch.Generator(device=device).manual_seed(1982)

        def z(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        attended = (
            torch.randn(
                (rows, 2, 4096),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.125
        )
        residual = torch.randn(
            (rows, 4, 5120), device=device, dtype=torch.bfloat16, generator=generator
        )
        post = torch.ones((rows, 4), device=device, dtype=torch.float32)
        comb = (
            torch.eye(4, device=device, dtype=torch.float32)
            .expand(rows, 4, 4)
            .contiguous()
        )
        pre = torch.full((rows, 4), 0.25, device=device, dtype=torch.float32)
        hc = MHCScratch.allocate(rows, 5120, device)
        shared = MLPScratch.allocate(rows, 5120, 576, device)
        routes = RouterScratch.allocate(rows, device, experts=384, top_k=6)
        partial, attention_out, routed, mlp_partial, mlp_output = [
            z(rows, 5120) for _ in range(5)
        ]
        logits = z(rows, 384, dtype=torch.float32)
        next_pre = z(rows, 4, dtype=torch.float32)
        with RecordedPreparationSession(
            device=device,
            checkpoint=str(checkpoint),
            workload_schema="persistent-decoder-checkpoint-parity",
            numerical_policy="bf16-seams",
            coordinator=coordinator,
            autotune_effort="off",
        ) as session:
            session.register_scratch(arena)
            projection = PreparedGroupedOutputProjection(
                attended,
                weights.attention["output"],
                partial,
                session=session,
                name="wo",
                workspace=arena,
            )
            attention_reduce = PreparedAllReduce(
                partial,
                attention_out,
                runtime=runtime,
                session=session,
                coordinator=coordinator,
                name="attention_reduce",
            )
            r = weights.residual
            ffn_hc = PreparedMHC(
                (attention_out, residual, post, comb),
                MHCWeights(
                    r["hc_ffn_fn"],
                    r["hc_ffn_scale"],
                    r["hc_ffn_base"],
                    r["ffn_norm.weight"],
                ),
                hc,
                session=session,
                name="ffn_hc",
                workspace=arena,
                pre_mix=pre,
                pre_out=next_pre,
                rms_eps=text["rms_norm_eps"],
                norm_eps=text["rms_norm_eps"],
                hc_eps=text["hc_eps"],
                sinkhorn_iters=text["hc_sinkhorn_iters"],
            )
            router_projection = UnquantizedProjection(
                hc.normalized,
                weights.experts.router,
                logits,
                session=session,
                name="router_projection",
            )
            router = PreparedRouter(
                logits,
                routes,
                session=session,
                name="route",
                score_func=text["scoring_func"],
                renormalize=text["norm_topk_prob"],
                routed_scaling_factor=text["routed_scaling_factor"],
                correction_bias=weights.experts.correction,
            )
            experts = PreparedRoutedExperts(
                hc.normalized,
                weights.experts.experts,
                routes.ids,
                routes.weights,
                routed,
                session=session,
                name="experts",
                workspace=arena,
                router_plan=router.plan,
            )
            mlp = PreparedMLP(
                hc.normalized,
                weights.experts.shared,
                shared.gate_up,
                shared.intermediate,
                shared.output,
                session=session,
                name="shared",
                workspace=arena,
                swiglu_limit=text["swiglu_limit"],
                round_silu=False,
            )
            mlp_reduce = PreparedAllReduce(
                mlp_partial,
                mlp_output,
                runtime=runtime,
                session=session,
                coordinator=coordinator,
                name="mlp_reduce",
            )
            if rows <= 8:
                prologue_weights = PersistentPrologueWeights.pack(weights)
                pro_hc = MHCScratch.allocate(rows, 5120, device)
                pro_pre = z(rows, 4, dtype=torch.float32)
                previous_y = (
                    torch.randn(
                        (rows, 5120),
                        device=device,
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                    * 0.125
                )
                pro_mhc = PreparedMHC(
                    (previous_y, residual, post, comb),
                    MHCWeights(
                        r["hc_attn_fn"],
                        r["hc_attn_scale"],
                        r["hc_attn_base"],
                        r["attn_norm.weight"],
                    ),
                    pro_hc,
                    session=session,
                    name="pro_mhc",
                    workspace=arena,
                    pre_mix=pre,
                    pre_out=pro_pre,
                    rms_eps=text["rms_norm_eps"],
                    norm_eps=text["rms_norm_eps"],
                    hc_eps=text["hc_eps"],
                    sinkhorn_iters=text["hc_sinkhorn_iters"],
                )
                projected = z(rows, 1792)
                query_low = z(rows, 1280)
                kv = z(rows, 512)
                query = z(rows, 16, 512)
                qkv_projection = QuantizedProjection(
                    pro_hc.normalized,
                    weights.attention["qkv_a"],
                    projected,
                    session=session,
                    name="pro_qkv",
                    workspace=arena,
                    mxfp8_min_amax=1e-4,
                )
                split_norm = PreparedMLANormQuant(
                    projected,
                    (weights.attention["q_norm"], weights.attention["kv_norm"]),
                    query_low,
                    kv,
                    eps=1e-20,
                    session=session,
                    name="pro_norm",
                )
                query_projection = QuantizedProjection(
                    query_low,
                    weights.attention["q_b"],
                    query.view(rows, -1),
                    session=session,
                    name="pro_query",
                    workspace=arena,
                    encoded=split_norm.encoded,
                )
                positions = torch.arange(
                    512, 512 + rows, dtype=torch.int64, device=device
                )
                rotary = CompressedRotary(config, device)
                frequency = rotary.frequencies[bool(text["compress_ratios"][layer])]
                theta = positions.float()[:, None] * frequency[None, :]
                angles = torch.stack((theta.cos(), theta.sin()), dim=-1).contiguous()
            session.tune(show_progress=False)
            if rows <= 8:
                pro_mhc(previous_y, residual, post, comb)
                qkv_projection(pro_hc.normalized)
                split_norm(projected)
                query_projection(query_low)
                rotary.apply(
                    query, positions, compressed=bool(text["compress_ratios"][layer])
                )
                rotary.apply(
                    kv, positions, compressed=bool(text["compress_ratios"][layer])
                )
                torch.cuda.synchronize()
                expected_pro = [
                    t.clone()
                    for t in (
                        pro_hc.residual,
                        pro_hc.normalized,
                        query_low,
                        kv,
                        query,
                        pro_hc.post,
                        pro_hc.comb,
                        pro_pre,
                    )
                ]
                pro_io = [
                    previous_y,
                    residual,
                    post,
                    comb,
                    pre,
                    positions,
                    angles,
                    z(rows, 4, 5120),
                    z(rows, 5120),
                    z(rows, 1280),
                    z(rows, 512),
                    z(rows, 16, 512),
                    z(rows, 4, dtype=torch.float32),
                    z(rows, 4, 4, dtype=torch.float32),
                    z(rows, 4, dtype=torch.float32),
                ]
                channel.prologue(pro_io, list(prologue_weights.tensors))
                torch.cuda.synchronize()
                for name, actual, reference in zip(
                    (
                        "residual",
                        "normalized",
                        "query_low",
                        "kv",
                        "query",
                        "post",
                        "comb",
                        "pre",
                    ),
                    pro_io[7:],
                    expected_pro,
                ):
                    delta = actual.float() - reference.float()
                    relative = float(
                        delta.square().mean().sqrt()
                        / reference.float().square().mean().sqrt().clamp_min(1e-20)
                    )
                    print(
                        "rank",
                        rank,
                        "prologue",
                        name,
                        "relative_rms",
                        relative,
                        "max_abs",
                        float(delta.abs().max()),
                        flush=True,
                    )
                    assert torch.isfinite(actual).all()
                    assert relative < 0.02, (name, relative)

            def baseline():
                projection(attended)
                attention_reduce(partial)
                ffn_hc(attention_out, residual, post, comb)
                router(router_projection(hc.normalized))
                experts()
                mlp(hc.normalized)
                torch.add(routed, shared.output, out=mlp_partial)
                mlp_reduce(mlp_partial)

            baseline()
            torch.cuda.synchronize()
            expected = [
                t.clone() for t in (mlp_output, hc.residual, hc.post, hc.comb, next_pre)
            ]
            normalized_expected = hc.normalized.clone()
            attention_expected = attention_out.clone()
            routes_expected = routes.weights.clone()
            io = [
                attended,
                z(rows, dtype=torch.int64),
                z(rows, dtype=torch.int64),
                z(1, 64, dtype=torch.float32),
                residual,
                post,
                comb,
                pre,
                z(rows, 5120),
                z(rows, 4, 5120),
                z(rows, 4, dtype=torch.float32),
                z(rows, 4, 4, dtype=torch.float32),
                z(rows, 4, dtype=torch.float32),
            ]
            for workers in (1, 2):
                channel.run(
                    io,
                    list(packed.tensors),
                    blocks_per_sm=workers,
                    shared_a16=(
                        int(mlp.up_projection.plan.selection.config.mode == "a16")
                        + 2
                        * int(mlp.down_projection.plan.selection.config.mode == "a16")
                    ),
                )
                torch.cuda.synchronize()
                for name, actual, reference in zip(
                    ("ffn", "residual", "post", "comb", "pre"), io[8:], expected
                ):
                    delta = actual.float() - reference.float()
                    rms = float(delta.square().mean().sqrt())
                    rrms = float(reference.float().square().mean().sqrt())
                    cosine = float(
                        torch.nn.functional.cosine_similarity(
                            actual.float().reshape(1, -1),
                            reference.float().reshape(1, -1),
                        )
                    )
                    print(
                        "rank",
                        rank,
                        "workers",
                        workers,
                        "field",
                        name,
                        "max_abs",
                        float(delta.abs().max()),
                        "relative_rms",
                        rms / max(rrms, 1e-20),
                        "cosine",
                        cosine,
                        flush=True,
                    )
                    assert torch.isfinite(actual).all()
                    # Diagnose every intermediate before the final full-output assertion.
                    pass
                for name, actual, reference in [
                    (
                        "normalized",
                        channel.scratch[12]
                        .view(torch.bfloat16)[: rows * 5120]
                        .view(rows, 5120),
                        normalized_expected,
                    ),
                    (
                        "attn",
                        channel.scratch[7]
                        .view(torch.bfloat16)[: rows * 5120]
                        .view(rows, 5120),
                        attention_expected,
                    ),
                    (
                        "route_weights",
                        channel.scratch[16]
                        .view(torch.float32)[: rows * 6]
                        .view(rows, 6),
                        routes_expected,
                    ),
                ]:
                    delta = actual.float() - reference.float()
                    print(
                        "rank",
                        rank,
                        "workers",
                        workers,
                        "intermediate",
                        name,
                        "rms",
                        float(actual.float().square().mean().sqrt()),
                        "reference_rms",
                        float(reference.float().square().mean().sqrt()),
                        "delta_rms",
                        float(delta.square().mean().sqrt()),
                        flush=True,
                    )
            assert (
                torch.linalg.vector_norm(io[8].float() - expected[0].float())
                / torch.linalg.vector_norm(expected[0].float())
                < 0.02
            )
        channel.close()
        channel = None
        runtime.close()
        runtime = None
    finally:
        if channel is not None:
            channel.close()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    main(args.checkpoint, args.rows, args.layer)
