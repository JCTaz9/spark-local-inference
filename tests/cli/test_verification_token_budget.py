"""Serve-only exact packed capacities reach every rank before graph capture."""

import pickle
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.cli import _parser, display
from ignition.cli.preparation import speculation
from ignition.errors import UsageError
from ignition.recipes import resolve
from ignition.recipes.deepseek41.recipe import select_attention_projection_policy


def plan_for(*flags, policy="reference-gemv"):
    args = _parser().parse_args(["serve", "deepseek-v4.1-flash", "--max-concurrency", "1",
        "--mtp-draft-tokens", "7", "--prefill-chunk-size", "4096", "--speculative-token-capacity", "8",
        "--attention-projection-policy", policy, *flags])
    model = select_attention_projection_policy(resolve(args.model), policy)
    gpus = tuple(planning.Gpu(i, "SM120", f"GPU-{i}", 96 << 30, 95 << 30, (12, 0)) for i in range(4))
    plan = planning.fit(model.recipe, model.checkpoint, gpus, concurrency=args.max_concurrency,
        prefill_chunk=args.prefill_chunk_size, reserve_bytes=0, speculation=speculation(args, model))
    return args, model, plan


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("policy", ["auto", "reference-gemv"])
def test_serve_exact_budget_preserves_full_proposal_and_reaches_every_rank(monkeypatch, remote, policy):
    from ignition.preparation import rank_group
    from ignition.preparation.rebuild_manifest import from_plan

    args, model, plan = plan_for("--mtp-verification-token-budget", "8",
        "--draft-head-quantization", "mxfp8", policy=policy)
    assert plan.speculation.draft_tokens == 7 and plan.speculation.adaptive
    assert plan.to_dict()["speculation"]["verification_token_budget"] == 8
    assert "exact verification budget 8 tokens" in display.speculation(plan.speculation)
    ranks = SimpleNamespace(hosts=("a", "b", "c", "d")) if remote else None
    calls = []
    monkeypatch.setattr(rank_group, "prepare_group", lambda *a, **kw: calls.append((a, kw)))
    model.recipe.prepare("checkpoint", plan, compile_forwards=True, kernel_cache=None,
                         progress=None, ranks=ranks, package_storage=object())
    config = calls[0][0][1]
    for _ in range(plan.world_size):
        peer = pickle.loads(pickle.dumps(config))
        assert peer.packed_speculative_shapes == ((1, 8),)
        assert peer.speculative_shapes == ((1, 1),)
        assert peer.draft_tokens == 7 and peer.decode_tokens == 8
        assert peer.prefill_chunk_size == 4096
        assert peer.mla_compute_mode == "bf16"
        assert peer.attention_projection_policy == policy
        assert peer.draft_head_quantization == "mxfp8"
        assert peer.markov_head_quantization is None
        assert peer.stream_overlap and not peer.native_dense_activations
        assert peer.devices == ((0, 0, 0, 0) if remote else (0, 1, 2, 3))
    with pytest.raises(ValueError, match="cannot be exported"):
        from_plan(model, plan, compiled_forwards=True)


def test_omitted_budget_keeps_the_complete_catalog_and_plan_metadata():
    _, _, plan = plan_for()
    assert plan.speculation.verification_token_budget is None
    assert "verification_token_budget" not in plan.to_dict()["speculation"]
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    config = PreparationConfig(max_active=1, speculative_widths=tuple(range(1, 9)))
    assert config.packed_speculative_shapes == tuple((1, n) for n in range(2, 9))


@pytest.mark.parametrize("flags", [
    ("--mtp-verification-token-budget", "0"),
    ("--mtp-verification-token-budget", "1"),
    ("--mtp-verification-token-budget", "-1"),
    ("--mtp-verification-token-budget", "8", "--no-mtp"),
    ("--mtp-verification-token-budget", "8", "--mtp-draft-tokens", "0"),
    ("--mtp-verification-token-budget", "8", "--no-mtp-adaptive-verification"),
])
def test_invalid_or_disabled_packed_budget_is_rejected_before_preparation(flags):
    with pytest.raises(UsageError):
        plan_for(*flags)


def test_other_recipes_and_export_reject_the_serve_only_budget():
    args = _parser().parse_args(["serve", "qwen3.8-flash", "--mtp-verification-token-budget", "8"])
    with pytest.raises(UsageError, match="DeepSeek V4.1"):
        speculation(args, resolve(args.model))
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["export", "deepseek-v4.1-flash", "--mtp-verification-token-budget", "8"])
    assert error.value.code == 2
    _, _, plan = plan_for("--mtp-verification-token-budget", "8")
    other = resolve("qwen3.8-flash")
    with pytest.raises(UsageError, match="DeepSeek V4.1"):
        planning.fit(other.recipe, other.checkpoint, plan.gpus, concurrency=1, reserve_bytes=0,
                     speculation=plan.speculation)
