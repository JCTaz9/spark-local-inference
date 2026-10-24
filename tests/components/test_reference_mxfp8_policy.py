"""Explicit attention policy selection preserves local checkpoint identity."""

import json
from dataclasses import replace

import pytest


def _csf_model(tmp_path):
    from ignition.recipes import resolve

    source = tmp_path / "flare-ds41-1017-csf"
    (source / "metadata").mkdir(parents=True)
    (source / "metadata" / "config.json").write_text(
        json.dumps({"architectures": ["DeepseekV41ForCausalLM"]})
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lil-mxfp4-csf-checkpoint/1",
                "compressed_weight_file_bytes": 500_000_000_000,
            }
        )
    )
    return resolve(str(source))


def test_reference_policy_preserves_local_csf_checkpoint_and_plan(
    tmp_path, monkeypatch
):
    from ignition import planning
    from ignition.recipes import aliases
    from ignition.recipes.deepseek41.recipe import (
        RECIPE,
        REFERENCE_GEMV_RECIPE,
        select_attention_projection_policy,
    )
    from ignition.preparation.rebuild_manifest import from_plan

    model = _csf_model(tmp_path)
    original = model.checkpoint
    before = (model.directory / "manifest.json").read_bytes()
    selected = select_attention_projection_policy(model, "reference-gemv")
    assert model.recipe is RECIPE and selected.recipe is REFERENCE_GEMV_RECIPE
    assert selected.checkpoint is original and selected.directory is model.directory
    assert selected.model_id == model.model_id
    assert (model.directory / "manifest.json").read_bytes() == before
    assert "deepseek-v4.1-flash-reference-gemv" not in aliases()
    assert selected.recipe.b12x_environment == RECIPE.b12x_environment
    assert "B12X_DENSE_SPLITK_TURBO" not in RECIPE.b12x_environment
    assert selected.recipe.draft_head_quantizations == ("mxfp8",)
    assert selected.recipe.markov_head_quantizations == ()
    gpus = tuple(
        planning.Gpu(i, "SM120", f"GPU-{i}", 96 << 30, 95 << 30, (12, 0))
        for i in range(4)
    )
    plan = planning.fit(
        selected.recipe,
        selected.checkpoint,
        gpus,
        concurrency=32,
        reserve_bytes=0,
        speculation=selected.recipe.speculation,
    )
    assert plan.checkpoint is original
    assert plan.attention_projection_policy == "reference-gemv"
    assert plan.to_dict()["attention_projection_policy"] == "reference-gemv"
    assert plan.mla_compute_mode == "bf16"
    from ignition.preparation import source_identity

    monkeypatch.setattr(source_identity, "source_identity", lambda: "source-test")
    rebuilt = from_plan(selected, plan, compiled_forwards=True)
    assert rebuilt["recipe"] == "deepseek41-reference-gemv"
    assert rebuilt["local_checkpoint"]
    assert rebuilt["checkpoint_repository"] == original.repository
    assert rebuilt["checkpoint_revision"] == ""


def test_explicit_auto_overrides_internal_reference_default(tmp_path):
    from ignition.recipes.deepseek41.recipe import (
        RECIPE,
        REFERENCE_GEMV_RECIPE,
        select_attention_projection_policy,
    )

    model = replace(_csf_model(tmp_path), recipe=REFERENCE_GEMV_RECIPE)
    assert select_attention_projection_policy(model, None) is model
    selected = select_attention_projection_policy(model, "auto")
    assert selected.recipe is RECIPE
    assert (
        selected.checkpoint is model.checkpoint
        and selected.directory is model.directory
    )


@pytest.mark.parametrize("policy", ["auto", "reference-gemv"])
def test_policy_flag_rejects_other_recipes(policy):
    from ignition.errors import UsageError
    from ignition.recipes import resolve
    from ignition.recipes.deepseek41.recipe import select_attention_projection_policy

    with pytest.raises(UsageError, match="only for DeepSeek V4.1"):
        select_attention_projection_policy(resolve("qwen3.8-flash"), policy)


def test_reference_config_requires_bf16_mla_and_explicit_draft_precision():
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    assert PreparationConfig().attention_projection_policy == "auto"
    reference = PreparationConfig(
        attention_projection_policy="reference-gemv", mla_compute_mode="bf16"
    )
    assert reference.attention_projection_policy == "reference-gemv"
    draft = replace(reference, draft_head_quantization="mxfp8")
    assert draft.draft_head_quantization == "mxfp8"
    assert draft.markov_head_quantization is None
    assert reference.draft_head_quantization is None
    with pytest.raises(ValueError, match="policy"):
        PreparationConfig(attention_projection_policy="unknown")
    with pytest.raises(ValueError, match="BF16 MLA"):
        PreparationConfig(
            attention_projection_policy="reference-gemv", mla_compute_mode="auto"
        )
    with pytest.raises(ValueError, match="checkpoint heads"):
        PreparationConfig(
            attention_projection_policy="reference-gemv",
            mla_compute_mode="bf16",
            draft_head_quantization="nvfp4",
        )
    with pytest.raises(ValueError, match="checkpoint heads"):
        replace(reference, markov_head_quantization="nvfp4")


def test_reference_policy_reaches_both_forward_factories():
    from ignition.recipes.deepseek41.bootstrap import _forward_factories

    decoder, draft = _forward_factories(
        "bf16",
        window_staging=None,
        suffix_scratch={},
        vision=None,
        attention_projection_policy="reference-gemv",
    )
    assert decoder.keywords["attention_projection_policy"] == "reference-gemv"
    assert draft.keywords["attention_projection_policy"] == "reference-gemv"


def test_cli_resolver_applies_policy_before_verification_without_source_replacement(
    tmp_path, monkeypatch
):
    from ignition.cli.config import Configuration
    from ignition.cli.preparation import resolve_model
    from ignition.recipes import ResolvedModel

    original = _csf_model(tmp_path)
    verified = []
    monkeypatch.setattr(
        ResolvedModel, "verify", lambda model, hub: verified.append(model)
    )
    config = Configuration(tmp_path / "absent.toml", {}, False)
    selected = resolve_model(
        config, str(original.directory), attention_projection_policy="reference-gemv"
    )
    assert (
        selected.checkpoint == original.checkpoint
        and selected.directory == original.directory
    )
    assert selected.recipe.attention_projection_policy == "reference-gemv"
    assert verified == [selected]
