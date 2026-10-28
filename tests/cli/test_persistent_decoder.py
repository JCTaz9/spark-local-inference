"""Persistent decoder intent, fixed storage budget and typed rebuild settings."""

from dataclasses import replace

import pytest

from ignition import planning
from ignition.cli import _parser
from ignition.errors import UsageError
from ignition.preparation import runtime_package_pb2 as proto
from ignition.recipes import resolve


def _plan(
    *,
    enabled=False,
    count=4,
    capability=(12, 0),
    scales="native",
    model="deepseek-v4.1-flash",
):
    selected = resolve(model)
    gpus = tuple(
        planning.Gpu(
            i,
            "NVIDIA RTX PRO 6000 Blackwell",
            f"GPU-{i}",
            96 << 30,
            95 << 30,
            capability,
        )
        for i in range(count)
    )
    return planning.plan(
        selected.recipe,
        selected.checkpoint,
        gpus,
        selected=tuple(range(count)),
        concurrency=1,
        reserve_bytes=0,
        speculation=selected.recipe.speculation,
        expert_scale_compression=scales,
        persistent_decoder=enabled,
    )


def test_flag_defaults_and_explicit_preparation_intent():
    for command in ("export", "debug-serve"):
        assert not _parser().parse_args([command]).persistent_decoder
        assert (
            _parser().parse_args([command, "--persistent-decoder"]).persistent_decoder
        )


def test_persistent_storage_reduces_available_cache_budget():
    baseline, persistent = _plan(), _plan(enabled=True)
    assert persistent.persistent_decoder and persistent.to_dict()["persistent_decoder"]
    assert persistent.fixed_bytes > baseline.fixed_bytes
    assert persistent.pages < baseline.pages
    assert replace(persistent, persistent_decoder=False).persistent_decoder is False


@pytest.mark.parametrize(
    "settings,reason",
    [
        ({"count": 3}, "requires DeepSeek V4.1 and four SM120 GPUs"),
        ({"capability": (10, 3)}, "requires DeepSeek V4.1 and four SM120 GPUs"),
        ({"scales": "csf"}, "requires --expert-scale-compression native"),
    ],
)
def test_unsupported_persistent_storage_or_geometry_is_rejected(settings, reason):
    with pytest.raises(UsageError, match=reason):
        _plan(enabled=True, **settings)


def test_rebuild_switch_has_an_explicit_false_default():
    legacy = proto.RebuildManifest.FromString(
        proto.RebuildManifest().SerializeToString()
    )
    assert not legacy.persistent_decoder
    prepared = proto.RebuildManifest(persistent_decoder=True)
    assert proto.RebuildManifest.FromString(
        prepared.SerializeToString()
    ).persistent_decoder


def test_upstream_joint_tuning_wire_tag_does_not_enable_persistent_decoder():
    # Upstream preparation metadata uses field 23 for joint attention-plan
    # tuning. Its bool payload must not select a persistent decoder here.
    upstream = bytes.fromhex("08 01 b8 01 01")
    decoded = proto.RebuildManifest.FromString(upstream)
    assert decoded.version == 1 and not decoded.persistent_decoder
    assert decoded.SerializeToString() == upstream


def test_persistent_decoder_uses_wire_tag26_and_preserves_vocabulary_sampling_tag11():
    encoded = bytes.fromhex("08 01 92 01 02 58 01 d0 01 01")
    decoded = proto.RebuildManifest.FromString(encoded)
    assert decoded.version == 1 and decoded.persistent_decoder
    assert decoded.speculation.vocabulary_parallel_sampling
    assert decoded.SerializeToString() == encoded
    prepared = proto.RebuildManifest(version=1, persistent_decoder=True,
        speculation=proto.RebuildSpeculation(vocabulary_parallel_sampling=True))
    assert prepared.SerializeToString() == encoded
    assert prepared.speculation.SerializeToString() == bytes.fromhex("58 01")
