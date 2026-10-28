"""Qualify and race complete checkpoint layer work through prepared compositions."""

from datetime import timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from ignition.recipes.deepseek41.contextual_tuning import CatalogLayerFixture


class Fixture(CatalogLayerFixture):
    """Use catalog input restoration and inspect one routed expert's packed weights."""

    def validate_numeric(self, layer):
        from ignition.recipes.deepseek41.contextual_tuning import CatalogLayerFixture

        if not getattr(self, "anchored", False):
            from ignition.recipes.deepseek41.persistent_moe_reference import (
                decode_weight,
            )

            self.anchored = True
            expert = int(layer.scratch.router.ids.flatten()[0])
            rep = layer.weights.experts.experts._impl.representation_for("w4a8_mx")
            for name, values, scales, rows, columns, group, coords in (
                (
                    "w13",
                    rep.w13_rp,
                    rep.w13_sfb,
                    1152,
                    5120,
                    576,
                    (
                        (0, 0),
                        (575, 5119),
                        (576, 0),
                        (1151, 5119),
                        (512, 32),
                        (1088, 4096),
                    ),
                ),
                (
                    "w2",
                    rep.w2_rp,
                    rep.w2_sfb,
                    5120,
                    576,
                    5120,
                    ((0, 0), (127, 511), (128, 512), (5119, 575)),
                ),
            ):
                decoded = decode_weight(
                    values[expert],
                    scales[expert],
                    rows=rows,
                    columns=columns,
                    group_rows=group,
                )
                print(
                    "SOURCE_ANCHOR",
                    dist.get_rank(),
                    expert,
                    name,
                    [(r, c, float(decoded[r, c])) for r, c in coords],
                    flush=True,
                )
                del decoded
        return CatalogLayerFixture.validate_numeric(self, layer)


class Selection:
    def __init__(self, fixture, selector):
        self.fixture, self.selector = fixture, selector

    def finalist_policies(self, requests):
        return {}

    def __call__(self, session, requests, finalists):
        session.preview_context_bindings()
        report = self.selector._race(session, self.fixture, self.fixture.layer)
        return {}, report


def _rank(rank, checkpoint, root, carry):
    from types import SimpleNamespace
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.deepseek41.attention import Rotary
    from ignition.recipes.deepseek41.decoder import PreparedLayer, Scratch
    from ignition.recipes.deepseek41.pool import ContinuationPool
    from ignition.recipes.deepseek41.persistent_selection import PersistentSelection
    from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=(root / "gloo").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=10),
    )
    runtime = selector = None
    try:
        with PackageStorage(rank=rank, world=4, device=rank) as storage:
            manifest = inspect_indexed_checkpoint(checkpoint)
            with DeepSeekWeightLoader(
                manifest,
                device,
                shard=TensorParallelShard(rank, 4),
                prefixes=("layers.0.", "embed.weight"),
            ) as loader:
                weights = loader.load_layer(0)
                # Catalog restoration samples token IDs across the checkpoint vocabulary.
                embedding = loader.load_tensor("embed.weight")
                config = loader.config
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(
                dist.group.WORLD,
                device,
                max_tokens=8,
                hidden_size=5120,
                ipc=storage.peer_storage,
            )
            arena = ScratchArena(device)
            pool = ContinuationPool.allocate(
                config,
                2,
                33,
                device,
                storage=storage,
                maximum_pages=64,
                maximum_rows=2,
                backed_slots=2,
            )
            metadata = Metadata.allocate(8, 1, 64, device)
            scratch = Scratch.allocate(8, 8, 4, device)
            signature = GraphSignature(
                "speculative", 1, 8, packed_token_capacity=8, draft_tokens=7
            )
            with RecordedPreparationSession(
                device=device,
                checkpoint=str(checkpoint),
                workload_schema="persistent-checkpoint-layer-selection",
                numerical_policy="bf16-seams",
                coordinator=coordinator,
                compile_workers=4,
                headroom_bytes=1 << 30,
                autotune_effort="off",
            ) as session:
                session.register_scratch(arena)
                layer = PreparedLayer(
                    0,
                    weights,
                    None,
                    scratch,
                    metadata,
                    pool,
                    Rotary(config, device),
                    shard=TensorParallelShard(rank, 4),
                    mode="verify",
                    runtime=runtime,
                    coordinator=coordinator,
                    session=session,
                    name="checkpoint.layer0",
                    workspace=arena,
                    carry=carry,
                    defer_post=False,
                )
                decoder = SimpleNamespace(layers=(layer,),
                    weights=SimpleNamespace(config=config, embedding=embedding), vision=None)
                fixture = Fixture("checkpoint.target", signature, decoder, 0)
                selector = PersistentSelection(
                    {"checkpoint": SimpleNamespace(target=SimpleNamespace(decoder=decoder))},
                    {"checkpoint": signature},
                    ordinary=None,
                    arena=arena,
                    coordinator=coordinator,
                    collective=runtime,
                    device=device,
                    graph_pool=storage.graph_pool.id,
                    cache_root=root / "kernels",
                    compile_decoder=True,
                    warmup_storage=storage,
                )
                report = session.tune(
                    contextual=Selection(fixture, selector), show_progress=False
                )
                print(
                    "rank",
                    rank,
                    "complete checkpoint layer selection",
                    report["contextual_tuning"],
                    flush=True,
                )
            selector.close()
            selector = None
            runtime.close()
            runtime = None
    finally:
        if selector is not None:
            selector.close()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def main(checkpoint, root, carry=False):
    root.mkdir()
    mp.spawn(_rank, args=(checkpoint, root, carry), nprocs=4, join=True)


if __name__ == "__main__":
    main(
        Path(sys.argv[1]).resolve(),
        Path(sys.argv[2]).resolve(),
        "--carry" in sys.argv[3:],
    )
