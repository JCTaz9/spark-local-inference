"""Export prepared DMA reductions after plain and compiled Flare capture."""

from datetime import timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class Forward:
    def __init__(self, operation, source):
        self.operation, self.source = operation, source

    def __call__(self):
        self.operation(self.source)


def operands(rows, rank, step):
    # Quarter integers keep every ring partial and the final sum exact in BF16.
    values = (torch.arange(5120, dtype=torch.int32) % 7 - 3 + rank + step).float() * .25
    return values.bfloat16().expand(rows, -1).contiguous()


def expected(rows, step):
    result = sum((operands(1, rank, step).float() for rank in range(4)))
    return result.bfloat16().expand(rows, -1).contiguous()


def _rank(rank, directory):
    from collective_package_fixture import _configuration, _process
    from helios.tests.dma_capture_fixture import Forward
    from ignition.components.collectives import PreparedAllReduce
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation, create_dma_all_reduce_runtime
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=(directory / "gloo").as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(minutes=5))
    folder = directory / f"rank-{rank}"
    folder.mkdir()
    runtime = None
    retained_graphs, compiled_owners = [], []
    try:
        with ModuleImages() as images, PackageStorage(rank=rank, world=4, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_dma_all_reduce_runtime(dist.group.WORLD, device,
                max_tokens=4096, hidden_size=5120, ipc=storage.peer_storage)
            assert runtime._fp8 == "", "fixture requires the default lossless prefill transport"
            shapes = {"init": 1, "decode": 1, "prefill128": 128, "prefill4096": 4096,
                      "resume": 1, "restore": 1, "snapshot": 1}
            sources = {rows: torch.empty((rows, 5120), dtype=torch.bfloat16, device=device)
                       for rows in set(shapes.values())}
            outputs = {name: torch.empty_like(sources[rows]) for name, rows in shapes.items()}
            for rows, source in sources.items():
                source.copy_(operands(rows, rank, 0))
            with RecordedPreparationSession(device=device, checkpoint="prefill-dma-capture",
                    workload_schema="tp4-bf16-prefill-capture", numerical_policy="exact-dyadic-ring",
                    coordinator=coordinator, autotune_effort="off", compile_workers=4) as session:
                operations = {name: PreparedAllReduce(sources[rows], outputs[name], runtime=runtime,
                    session=session, coordinator=coordinator, name=name) for name, rows in shapes.items()}
                functions = {name: Forward(operations[name], sources[rows]) for name, rows in shapes.items()}
                signatures = {name: GraphSignature("prefill" if name.startswith("prefill") else name,
                    1, rows if name.startswith("prefill") else 1) for name, rows in shapes.items()}
                session.tune(show_progress=False)
                for operation in operations.values():
                    assert operation.plan.prepared.state.query.surface == "DmaAllReduce.all_reduce"
                stream = torch.cuda.Stream(device=device)
                catalogs = []
                for compile_decoder in (False, True):
                    graphs, compiled = capture_forwards(functions, signatures, session=session,
                        device=device, stream=stream, pool=storage.graph_pool.id,
                        collective=runtime, coordinator=coordinator, compile_decoder=compile_decoder,
                        warmup_storage=storage, cache_root=directory / "forwards", report_catalog=False,
                        retain_graphs=True)
                    retained_graphs.extend(graphs.values())
                    compiled_owners.append(compiled)
                    catalogs.append(graphs)
                    print(f"rank {rank}: {'compiled' if compile_decoder else 'plain'} DMA capture passed", flush=True)
                session.freeze()
                for graphs in catalogs:
                    for name in ("prefill4096", "prefill128", "decode", "prefill4096"):
                        rows = shapes[name]
                        for step in (0, 1):
                            sources[rows].copy_(operands(rows, rank, step))
                            outputs[name].fill_(float("nan"))
                            torch.cuda.synchronize(device)
                            dist.barrier()
                            before = torch.cuda.memory_stats(device)
                            graphs[name].replay()
                            torch.cuda.synchronize(device)
                            after = torch.cuda.memory_stats(device)
                            for key in ("allocation.all.allocated", "allocation.all.freed",
                                        "segment.all.allocated", "segment.all.freed"):
                                assert before[key] == after[key], key
                            assert torch.equal(outputs[name].cpu().view(torch.uint8),
                                               expected(rows, step).view(torch.uint8))
                            assert torch.equal(sources[rows].cpu(), operands(rows, rank, step))
                print(f"rank {rank}: frozen plain/compiled DMA replay and stable allocation passed", flush=True)
                for graph in catalogs[0].values():
                    graph.reset()
                state = storage.device_allocate("continuations", (4, 32), torch.uint8,
                    group="STATE_ROWS", maximum=4, units=4)
                pages = storage.device_allocate("pages", (128, 16), torch.uint8,
                    group="KV_PAGES", maximum=128, units=128)
                records = torch.zeros((2, 1, 24), dtype=torch.int64, device=device)
                records[..., 4] = -1
                records[..., 12:14] = -1
                records[..., 15] = 1
                records[..., 16:] = -1
                buffers = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                    "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                    "completion_records": records, "state_pool": state, "pages": pages,
                    "memory_capacity": torch.tensor([4, 128], device=device)}
                buffers.update({f"source-{rows}": value for rows, value in sources.items()})
                buffers.update({f"output-{name}": value for name, value in outputs.items()})
                prepared = PreparedExecution(device=rank, graphs=catalogs[1], buffers=buffers,
                    signatures=signatures, max_active=1,
                    resources=(runtime, session, operations, functions, sources, outputs,
                               storage, stream, compiled_owners))
                prepared.server_config = {**_configuration(folder, signatures),
                    "max_sequence_tokens": 8192, "prefill_chunk_size": 4096, "pages": 128}
                prepared.package_data = None
                cases = []
                for step in (0, 1):
                    for rows in sources:
                        (folder / f"source-{rows}-{step}.bin").write_bytes(
                            operands(rows, rank, step).view(torch.uint8).numpy().tobytes())
                    for name, rows in shapes.items():
                        path = f"expected-{name}-{step}.bin"
                        (folder / path).write_bytes(expected(rows, step).view(torch.uint8).numpy().tobytes())
                        cases.append(dict(graph=name, source=f"source-{rows}", output=f"output-{name}",
                            input=f"rank-{rank}/source-{rows}-{step}.bin",
                            expected=f"rank-{rank}/{path}", step=step))
                torch.cuda.synchronize(device)
                dist.barrier()
                storage.peer_storage.reset()
                torch.cuda.synchronize(device)
                dist.barrier()
                export_model(prepared, images, storage, directory / f"rank-{rank}.json",
                             options={}, bundle_only=rank != 0, sources=True)
                (folder / "metadata.json").write_text(json.dumps(dict(cases=cases, process=_process(),
                    owned=list(storage.peer_storage.owned.items()))))
                for graph in retained_graphs:
                    graph.reset()
                retained_graphs.clear()
                compiled_owners.clear()
                runtime.close()
                runtime = None
    finally:
        for graph in retained_graphs:
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def prepare(directory):
    from collective_package_fixture import _process
    from signing import publisher
    from ignition.preparation.package_file import combine

    directory.mkdir()
    mp.spawn(_rank, args=(directory,), nprocs=4, join=True)
    ranks = [json.loads((directory / f"rank-{rank}" / "metadata.json").read_text()) for rank in range(4)]
    peers = [dict(owner=rank, address=address, nbytes=size, readers=list(range(4)))
             for rank, metadata in enumerate(ranks) for address, size in metadata["owned"]]
    combine(directory / "package", [directory / f"rank-{rank}.json" for rank in range(4)],
            peers, publisher=publisher())
    (directory / "collective-expected.json").write_text(json.dumps(dict(ranks=ranks,
        preparation_processes=[_process(), *(rank["process"] for rank in ranks)])))
    print("exported DMA package after plain/compiled Flare capture and frozen replay", flush=True)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]).resolve())
