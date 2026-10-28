"""Share one compiled persistent decoder across TP4 rank-local bindings."""

from datetime import timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from persistent_decoder_package_fixture import fixture_io, zero_weights


class Forward:
    def __init__(self, call, prologue):
        self.call, self.prologue = call, prologue

    def __call__(self):
        self.prologue()
        self.call()


def _rank(rank, root):
    from flashinfer.experimental.persistent_decoder import PersistentDecoderTP4
    from ignition.components.persistent_decoder import (
        PersistentDecoderCall,
        PersistentDecoderWeights,
        PersistentPrologueCall,
        PersistentPrologueWeights,
    )
    from ignition.preparation.aot import aot_forward, BoundForward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.device_binding import bind_device

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=(root / "gloo").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=5),
    )
    channel = None
    graph = None
    try:
        with PackageStorage(rank=rank, world=4, device=rank) as storage:
            channel = PersistentDecoderTP4(
                dist.group.WORLD, device=device, ipc=storage.peer_storage
            )
            weights = PersistentDecoderWeights(tuple(zero_weights(device)), 0, None)
            source = torch.ones((1, 4, 5120), dtype=torch.bfloat16, device=device)
            io = fixture_io(source, device)
            # Distinct registrations require the shared artifact to bind each
            # rank's actual operation handle rather than restore a constant ID.
            _other_bindings = [
                PersistentDecoderCall.bind(
                    io, weights, channel, workers=2, shared_a16=0
                )
                for _ in range(rank)
            ]

            def z(*shape, dtype=torch.bfloat16):
                return torch.zeros(shape, dtype=dtype, device=device)

            def one(width):
                return torch.ones(width,dtype=torch.bfloat16,device=device)
            prologue_weights = PersistentPrologueWeights(
                (
                    z(24, 20480, dtype=torch.float32),
                    one(5120),
                    z(1792, 5120, dtype=torch.uint8),
                    z(224, 80, 4, 2, 2, dtype=torch.uint8),
                    one(1280),
                    one(512),
                    z(8192, 1280, dtype=torch.uint8),
                    z(1024, 20, 4, 2, 2, dtype=torch.uint8),
                    z(3, dtype=torch.float32),
                    z(24, dtype=torch.float32),
                ),
                0,
            )
            angles = z(1, 32, 2, dtype=torch.float32)
            angles[..., 0] = 1
            output_residual = z(1, 4, 5120)
            pro_io = [
                z(1, 5120),
                source,
                z(1, 4, dtype=torch.float32),
                torch.eye(4, dtype=torch.float32, device=device).view(1, 4, 4),
                torch.full((1, 4), 0.25, dtype=torch.float32, device=device),
                z(1, dtype=torch.int64),
                angles,
                output_residual,
                z(1, 5120),
                z(1, 1280),
                z(1, 512),
                z(1, 16, 512),
                io[5],
                io[6],
                io[7],
            ]
            io[4] = output_residual
            forward = Forward(
                PersistentDecoderCall.bind(
                    io, weights, channel, workers=2, shared_a16=0
                ),
                PersistentPrologueCall.bind(pro_io, prologue_weights, channel),
            )
            path = root / "compiled.bin"
            print("rank", rank, "begin compile/load", flush=True)
            if rank == 0:
                artifact = aot_forward(forward)
                artifact.save_compiled_function(str(path))
            dist.barrier()
            if rank != 0:
                with bind_device(device), path.open("rb") as stream:
                    artifact = torch.compiler.load_compiled_function(stream)
            print(
                "rank", rank, "loaded guards", artifact.guard_check(forward), flush=True
            )
            bound = BoundForward(artifact, forward, True)
            print("rank", rank, "before compiled execution", flush=True)
            bound()
            print("rank", rank, "compiled call returned", flush=True)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(io[9], source, atol=0, rtol=0)
            bound.prepare_capture()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=storage.graph_pool.id):
                bound()
            graph.instantiate()
            for value in (0.125, -0.25, 0.5):
                source.fill_(value)
                io[9].fill_(float("nan"))
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize(device)
                after = torch.cuda.memory_stats(device)
                torch.testing.assert_close(io[9], source, atol=0, rtol=0)
                for key in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[key] == after[key], (key, before[key], after[key])
            print(
                "rank",
                rank,
                "compiled shared artifact, local peer alias, runtime handle and stable replay passed",
                flush=True,
            )
            graph.reset()
            graph = None
            channel.close()
            channel = None
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if graph is not None:
            graph.reset()
        if channel is not None:
            channel.close()
        dist.destroy_process_group()


def main(root):
    root.mkdir()
    mp.spawn(_rank, args=(root,), nprocs=4, join=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
