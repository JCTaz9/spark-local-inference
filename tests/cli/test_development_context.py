"""Development serving backs the planned context without changing captured graphs."""

import json
import pickle
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.cli import _parser, serve
from ignition.cli.preparation import speculation
from ignition.errors import UsageError
from ignition.recipes import ResolvedModel, resolve
from ignition.recipes.deepseek41.recipe import select_attention_projection_policy


def context_plan(policy="reference-gemv"):
    args = _parser().parse_args(["serve", "deepseek-v4.1-flash", "--max-concurrency", "1",
        "--mtp-draft-tokens", "7", "--mtp-verification-token-budget", "8",
        "--speculative-token-capacity", "8", "--prefill-chunk-size", "4096",
        "--draft-head-quantization", "mxfp8", "--attention-projection-policy", policy])
    model = select_attention_projection_policy(resolve(args.model), policy)
    gpus = tuple(planning.Gpu(i, "SM120", f"GPU-{i}", 96 << 30, 95 << 30, (12, 0)) for i in range(4))
    plan = planning.fit(model.recipe, model.checkpoint, gpus, concurrency=1, prefill_chunk=4096,
        reserve_bytes=0, speculation=speculation(args, model))
    return args, model, plan


@pytest.mark.parametrize("policy", ["auto", "reference-gemv"])
def test_serve_resolves_full_context_before_preparation(monkeypatch, capsys, policy):
    import torch
    from ignition.preparation import package_storage, tuning_replicas

    args, model, base = context_plan(policy)
    monkeypatch.setattr(serve, "_resolve_preparation_model", lambda *a: model)
    monkeypatch.setattr(serve, "_address", lambda *a: "127.0.0.1:0")
    monkeypatch.setattr(ResolvedModel, "cached", lambda *a: True)
    monkeypatch.setattr(serve.Host, "discover", lambda: SimpleNamespace(gpus=base.gpus))
    monkeypatch.setattr(serve.environment, "apply", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(package_storage, "PackageStorage", lambda **kw: nullcontext())
    captured = []

    class PreparationReached(Exception):
        pass

    def prepare(*a, **kw):
        captured.append(a[2])
        raise PreparationReached

    monkeypatch.setattr(tuning_replicas, "prepare_recipe", prepare)
    with pytest.raises(PreparationReached):
        serve.serve(args)
    plan = captured[0]
    assert plan.context == base.context == 1048576
    assert plan.development_context_tokens == 1048576
    assert plan.development_kv_pages == 8192
    assert plan.prefill_chunk == 4096 and plan.speculation == base.speculation
    printed = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith('{"plan":')]
    assert printed[0]["plan"]["development_context_tokens"] == 1048576
    assert printed[0]["plan"]["development_kv_pages"] == 8192


@pytest.mark.parametrize("remote", [False, True])
def test_context_backing_reaches_every_rank_and_native_handoff(monkeypatch, remote):
    from ignition.preparation import rank_group
    from ignition.preparation.rebuild_manifest import from_plan

    _, model, base = context_plan()
    plan = replace(base, development_context_tokens=base.context)
    calls = []
    monkeypatch.setattr(rank_group, "prepare_group", lambda *a, **kw: calls.append((a, kw)))
    ranks = SimpleNamespace(hosts=("a", "b", "c", "d")) if remote else None
    model.recipe.prepare("checkpoint", plan, compile_forwards=True, kernel_cache=None,
        progress=None, ranks=ranks, package_storage=object())
    config = calls[0][0][1]
    for _ in range(plan.world_size):
        peer = pickle.loads(pickle.dumps(config))
        assert peer.kv_pages == 8192 and peer.context == 1048576
        assert peer.prefill_chunk_size == 4096 and peer.packed_speculative_shapes == ((1, 8),)
        assert peer.speculative_shapes == ((1, 1),) and peer.draft_tokens == 7
        assert peer.mla_compute_mode == "bf16" and peer.draft_head_quantization == "mxfp8"
    native = SimpleNamespace(max_active=1, server_config={"snapshots": 8, "pages": config.kv_pages,
        "model_contract": {"graphs": [{"state_rows_per_request": 8}]}},
        buffers={"memory_capacity": SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [10, 8192]))})
    assert serve._diagnostic_options(native, {}) == {"pages": 8192, "snapshots": 1}
    with pytest.raises(ValueError, match="cannot be exported"):
        from_plan(model, plan, compiled_forwards=True)


def test_export_preparation_retains_warmup_backing(monkeypatch):
    from ignition.preparation import rank_group

    _, model, plan = context_plan()
    calls = []
    monkeypatch.setattr(rank_group, "prepare_group", lambda *a, **kw: calls.append(a[1]))
    model.recipe.prepare("checkpoint", plan, compile_forwards=True, retain_graphs=True,
        kernel_cache=None, progress=None, package_storage=object())
    assert plan.development_context_tokens is None and plan.development_kv_pages is None
    assert calls[0].kv_pages is None


@pytest.mark.parametrize("tokens", [True, 0, -1, 4096, 1048577])
def test_invalid_programmatic_backing_is_rejected(tokens):
    _, _, plan = context_plan()
    with pytest.raises(UsageError, match="development context backing"):
        replace(plan, development_context_tokens=tokens)


def test_backing_respects_planned_memory_and_rounds_pages():
    _, _, plan = context_plan()
    assert replace(plan, development_context_tokens=131073).development_kv_pages == 1025
    small = replace(plan, pages=1024)
    assert replace(small, development_context_tokens=small.context).development_kv_pages == 1024
    with pytest.raises(UsageError, match="planned memory"):
        replace(small, development_context_tokens=1048576)
    other = resolve("qwen3.8-flash")
    with pytest.raises(UsageError, match="only DeepSeek"):
        replace(plan, recipe=other.recipe, development_context_tokens=131072)
