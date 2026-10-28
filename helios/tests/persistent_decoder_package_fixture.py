"""Export TP4 persistent decoder graphs for native replay after preparation exits."""

from datetime import timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collective_package_fixture import _configuration, _process
from signing import publisher


def zero_weights(device):
    """Alias expert rows because every fixture expert contains identical zeros."""

    def z(*shape, dtype=torch.uint8):
        return torch.zeros(shape, dtype=dtype, device=device)

    def repeated(n, fill=0):
        return torch.full((1, n), fill, dtype=torch.uint8, device=device).expand(
            384, -1
        )

    return [
        z(2048 * 4096),
        z(64, 128),
        z(5120 * 2048),
        z(160, 64),
        z(384 * 10240),
        -torch.arange(384, device=device, dtype=torch.float32),
        z(0, dtype=torch.float32),
        z(1152 * 5120),
        z(36, 160),
        z(5120 * 640),
        z(160, 20),
        repeated(576 * 5120),
        repeated(2 * 576 * 160, 127),
        repeated(5120 * 576 // 2),
        repeated(5120 * 18, 127),
        z(24, 20480, dtype=torch.float32),
        z(3, dtype=torch.float32),
        z(24, dtype=torch.float32),
        torch.ones(5120, device=device, dtype=torch.bfloat16),
    ]


def fixture_io(source, device):
    rows = source.shape[0]

    def z(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype, device=device)

    return [
        z(rows, 16, 512),
        z(rows, dtype=torch.int64),
        z(rows, dtype=torch.int64),
        z(1, 64, dtype=torch.float32),
        source,
        z(rows, 4, dtype=torch.float32),
        torch.eye(4, device=device, dtype=torch.float32)
        .expand(rows, 4, 4)
        .contiguous(),
        torch.full((rows, 4), 0.25, device=device, dtype=torch.float32),
        z(rows, 5120),
        z(rows, 4, 5120),
        z(rows, 4, dtype=torch.float32),
        z(rows, 4, 4, dtype=torch.float32),
        z(rows, 4, dtype=torch.float32),
        z(rows, 384, dtype=torch.float32),
        z(rows, 6, dtype=torch.int32),
        z(rows, 6, dtype=torch.float32),
        z(rows, 5120),
        z(rows, 5120),
        z(rows, 1152),
        z(rows, 576),
    ]


def _rank(rank, directory):
    from flashinfer.experimental.persistent_decoder import PersistentDecoderTP4
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "gloo").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=5),
    )
    folder = directory / f"rank-{rank}"
    folder.mkdir()
    channel = None
    captured = []
    try:
        with (
            ModuleImages() as images,
            PackageStorage(rank=rank, world=4, device=rank) as storage,
        ):
            channel = PersistentDecoderTP4(
                dist.group.WORLD, device=device, ipc=storage.peer_storage
            )
            weights = zero_weights(device)
            source = torch.ones((128, 4, 5120), dtype=torch.bfloat16, device=device)
            shapes = {
                "init": 1,
                "decode": 2,
                "prefill64": 17,
                "prefill128": 128,
                "restore": 1,
                "snapshot": 1,
                "resume": 1,
            }
            operands = {
                name: fixture_io(source[:rows], device) for name, rows in shapes.items()
            }
            prologues = {}

            def z(*shape, dtype=torch.bfloat16):
                return torch.zeros(shape, dtype=dtype, device=device)

            prologue_weights = [
                z(24, 20480, dtype=torch.float32),
                torch.ones(5120, device=device, dtype=torch.bfloat16),
                z(1792, 5120, dtype=torch.uint8),
                z(224, 80, 4, 2, 2, dtype=torch.uint8),
                torch.ones(1280, device=device, dtype=torch.bfloat16),
                torch.ones(512, device=device, dtype=torch.bfloat16),
                z(8192, 1280, dtype=torch.uint8),
                z(1024, 20, 4, 2, 2, dtype=torch.uint8),
                z(3, dtype=torch.float32),
                z(24, dtype=torch.float32),
            ]
            for name, rows in shapes.items():
                if rows > 8:
                    continue
                tail = operands[name]
                angles = z(rows, 32, 2, dtype=torch.float32)
                angles[..., 0] = 1
                residual_out = z(rows, 4, 5120)
                prologues[name] = [
                    z(rows, 5120),
                    source[:rows],
                    z(rows, 4, dtype=torch.float32),
                    torch.eye(4, device=device, dtype=torch.float32)
                    .expand(rows, 4, 4)
                    .contiguous(),
                    torch.full((rows, 4), 0.25, device=device, dtype=torch.float32),
                    z(rows, dtype=torch.int64),
                    angles,
                    residual_out,
                    z(rows, 5120),
                    z(rows, 1280),
                    z(rows, 512),
                    z(rows, 16, 512),
                    tail[5],
                    tail[6],
                    tail[7],
                ]
                tail[4] = residual_out

            def run(name):
                if name in prologues:
                    channel.prologue(prologues[name], prologue_weights)
                channel.run(operands[name], weights)

            stream = torch.cuda.Stream(device=device)
            graphs, signatures = {}, {}
            for name, io in operands.items():
                run(name)
                torch.cuda.synchronize(device)
                dist.barrier()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.graph(graph, stream=stream, pool=storage.graph_pool.id):
                    run(name)
                graph.instantiate()
                captured.append(graph)
                graphs[name] = graph
                signatures[name] = GraphSignature(
                    "prefill" if name.startswith("prefill") else name,
                    1,
                    int(name.removeprefix("prefill"))
                    if name.startswith("prefill")
                    else 1,
                )
            state = storage.device_allocate(
                "continuations",
                (4, 32),
                torch.uint8,
                group="STATE_ROWS",
                maximum=4,
                units=4,
            )
            pages = storage.device_allocate(
                "pages", (16, 16), torch.uint8, group="KV_PAGES", maximum=16, units=16
            )
            buffers = {
                "frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                "completion_records": torch.zeros(
                    (2, 1, 24), dtype=torch.int64, device=device
                ),
                "state_pool": state,
                "memory_capacity": torch.tensor([4, 16], device=device),
                "pages": pages,
                "source": source,
            }
            records = buffers["completion_records"]
            records[..., 4] = -1
            records[..., 12:14] = -1
            records[..., 15] = 1
            records[..., 16:] = -1
            buffers.update({f"output-{name}": io[9] for name, io in operands.items()})
            prepared = PreparedExecution(
                device=rank,
                graphs=graphs,
                buffers=buffers,
                signatures=signatures,
                max_active=1,
                resources=(
                    channel,
                    weights,
                    operands,
                    prologues,
                    prologue_weights,
                    storage,
                    stream,
                ),
            )
            prepared.server_config = _configuration(folder, signatures)
            prepared.package_data = None
            cases = []
            for step in range(3):
                generator = torch.Generator().manual_seed(813 + step)
                values = torch.randn(
                    (128, 4, 5120), dtype=torch.bfloat16, generator=generator
                )
                (folder / f"source-{step}.bin").write_bytes(
                    values.view(torch.uint8).numpy().tobytes()
                )
                source.copy_(values)
                for name, rows in shapes.items():
                    graphs[name].replay()
                    torch.cuda.synchronize(device)
                    path = f"expected-{name}-{step}.bin"
                    (folder / path).write_bytes(
                        operands[name][9].view(torch.uint8).cpu().numpy().tobytes()
                    )
                    cases.append(
                        dict(
                            graph=name,
                            source="source",
                            output=f"output-{name}",
                            input=f"rank-{rank}/source-{step}.bin",
                            expected=f"rank-{rank}/{path}",
                            step=step,
                        )
                    )
            torch.cuda.synchronize(device)
            dist.barrier()
            storage.peer_storage.reset()
            torch.cuda.synchronize(device)
            dist.barrier()
            export_model(
                prepared,
                images,
                storage,
                directory / f"rank-{rank}.json",
                options={},
                bundle_only=rank != 0,
                sources=True,
            )
            (folder / "metadata.json").write_text(
                json.dumps(
                    dict(
                        cases=cases,
                        process=_process(),
                        owned=list(storage.peer_storage.owned.items()),
                    )
                )
            )
            for graph in captured:
                graph.reset()
            captured.clear()
            channel.close()
            channel = None
    finally:
        for graph in captured:
            graph.reset()
        if channel is not None:
            channel.close()
        dist.destroy_process_group()


def prepare(directory):
    from ignition.preparation.package_file import combine

    directory.mkdir(exist_ok=True)
    mp.spawn(_rank, args=(directory,), nprocs=4, join=True)
    ranks = [
        json.loads((directory / f"rank-{rank}" / "metadata.json").read_text())
        for rank in range(4)
    ]
    peers = [
        dict(owner=rank, address=address, nbytes=size, readers=list(range(4)))
        for rank, metadata in enumerate(ranks)
        for address, size in metadata["owned"]
    ]
    combine(
        directory / "package",
        [directory / f"rank-{rank}.json" for rank in range(4)],
        peers,
        publisher=publisher(),
    )
    (directory / "collective-expected.json").write_text(
        json.dumps(
            dict(
                ranks=ranks,
                preparation_processes=[
                    _process(),
                    *(rank["process"] for rank in ranks),
                ],
            )
        )
    )
    print(
        "exported persistent decoder package after four preparation ranks exited",
        flush=True,
    )


if __name__ == "__main__":
    prepare(Path(sys.argv[1]).resolve())
