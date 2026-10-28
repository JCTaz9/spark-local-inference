"""Composed mHC planning keeps scratch, precision choices and weight ownership explicit."""

import gc
import weakref
from dataclasses import replace

import pytest
import torch

from ignition.components.mhc_sm103 import (
    Query,
    _layout,
    _plan_offset,
    _split_weights,
    _views,
    tactics,
)
from ignition.components.native_operator import NativeConfig


def query(**options):
    return Query(**({"rows": 513, "hidden": 4096, "operation": "post_pre", "rms_eps": 1e-6,
                    "hc_eps": 1e-6, "sinkhorn_iters": 20, "norm_eps": 1e-5,
                    "norm_dtype": "float32", "has_low": False, "source_identity": "test"} | options))


def aligned_storage(nbytes):
    allocation = torch.empty(nbytes + 256, dtype=torch.uint8)
    return allocation.narrow(0, (-allocation.data_ptr()) % 256, nbytes)


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
def test_composed_mhc_tactics_preserve_fp32_pipeline_restriction(hidden):
    q = query(hidden=hidden)
    assert {t[4] for t in tactics(q)} == {3, 4, 6}
    assert {t[4] for t in tactics(replace(q, has_low=True))} == {1}
    for tactic in tactics(q):
        assert (4 * hidden) % (tactic[1] * tactic[2]) == 0
        assert NativeConfig.from_config({"tactic": list(tactic)}).tactic == tactic


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("has_low", [False, True])
def test_two_token_finalizer_is_raced_for_hidden_4096_only(hidden, has_low):
    q = query(hidden=hidden, has_low=has_low)
    fused_pair = [t for t in tactics(q) if t[5] == "fused_pair"]
    if hidden != 4096:
        assert not fused_pair
        return
    assert fused_pair and {t[6] for t in fused_pair} == {160}
    # Every projection geometry races fused_pair beside the one-token finalizers.
    for geometry in {t[:5] for t in tactics(q)}:
        assert {t[5:] for t in tactics(q) if t[:5] == geometry} >= {
            ("separate", 256), ("separate", 128), ("fused", 160), ("fused_pair", 160),
            ("fused", 96), ("fused", 288)}
    for tactic in fused_pair:
        assert NativeConfig.from_config({"tactic": list(tactic)}).tactic == tactic
        assert _layout(q, tactic) == _layout(q, (*tactic[:5], "fused", 160))


def test_composed_mhc_selection_identity_changes_with_the_finalizer_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import flashinfer.jit.mhc

    from ignition.components import mhc_sm103

    finalizer = tmp_path / "mhc_pre_big_fuse.cu"
    finalizer.write_text("// finalizer body\n")
    spec = SimpleNamespace(sources=[finalizer], extra_cflags=None, extra_cuda_cflags=None,
                           extra_ldflags=None, extra_cuda_cflags_by_source=None)
    monkeypatch.setattr(flashinfer.jit.mhc, "gen_mhc_module", lambda: spec)
    compute = mhc_sm103._source_identity.__wrapped__
    first = compute()
    assert compute() == first
    finalizer.write_text("// finalizer body with two tokens per CTA\n")
    assert compute() != first
    finalizer.write_text("// finalizer body\n")
    assert compute() == first
    spec.extra_cuda_cflags = ["-use_fast_math"]
    assert compute() != first


def test_composed_mhc_scratch_uses_disjoint_aligned_regions_for_selected_tactic():
    q = query()
    for tactic in tactics(q):
        layout, nbytes = _layout(q, tactic)
        storage = aligned_storage(nbytes + 256)
        storage.fill_(91)
        views = _views(storage, q, tactic)
        assert len(views) == (3 if tactic[5] == "separate" else 2)
        for index, view in enumerate(views):
            assert view.data_ptr() % 256 == 0
            view.fill_(index + 1)
        for index, view in enumerate(views):
            assert torch.all(view == index + 1)
        assert torch.all(storage[nbytes:] == 91)
        for (offset, size, _, _), next_region in zip(layout, (*layout[1:], (nbytes,)), strict=True):
            assert torch.all(storage[offset + size:next_region[0]] == 91)
        with pytest.raises(ValueError, match="256-byte alignment"):
            _views(storage[1:], q, tactic)
        with pytest.raises(ValueError, match="byte capacity"):
            _views(storage[:nbytes - 1], q, tactic)


def test_composed_mhc_views_follow_the_plan_offset_inside_the_arena():
    q = query()
    for tactic in tactics(q):
        _, nbytes = _layout(q, tactic)
        arena = aligned_storage(4096 + nbytes + 256)
        for base in (0, 256, 4096):
            storage = arena.narrow(0, base, nbytes)
            offset = _plan_offset(arena, storage, nbytes)
            assert offset == base
            arena.fill_(91)
            views = _views(arena, q, tactic, offset)
            for index, view in enumerate(views):
                assert view.data_ptr() == storage.data_ptr() + _layout(q, tactic)[0][index][0]
                view.fill_(index + 1)
            assert torch.all(arena[:base] == 91)
            assert torch.all(arena[base + nbytes:] == 91)
        with pytest.raises(ValueError, match="byte capacity"):
            _views(arena, q, tactic, arena.nbytes - nbytes + 256)
        with pytest.raises(ValueError, match="outside its arena"):
            _plan_offset(arena[:nbytes], arena.narrow(0, 256, nbytes), nbytes)
        with pytest.raises(ValueError, match="smaller than its selected layout"):
            _plan_offset(arena, arena.narrow(0, 0, nbytes - 1), nbytes)
        with pytest.raises(ValueError, match="outside its arena"):
            _plan_offset(arena.narrow(0, 256, nbytes), arena, nbytes)


def test_split_weight_cache_releases_original_tensor_identity():
    fn = torch.randn(24, 16)
    high, low, has_low = _split_weights(fn)
    again = _split_weights(fn)
    assert again[0] is high and again[1] is low and has_low
    reference = weakref.ref(fn)
    del fn
    gc.collect()
    assert reference() is None
    # New immutable checkpoint data must get its own halves, even after the
    # original owner dies and its allocator storage becomes reusable.
    exact = torch.ones(24, 16)
    new_high, new_low, new_has_low = _split_weights(exact)
    assert new_high is not high and new_low is not low
    assert not new_has_low
    torch.testing.assert_close(new_high[:24].float(), exact, rtol=0, atol=0)


def test_split_weight_cache_shares_immutable_parameter_aliases_without_pinning_source():
    from ignition.components.mhc_sm103 import _SPLIT_WEIGHTS

    checkpoint = torch.randn(24, 16)
    first = torch.nn.Parameter(checkpoint, requires_grad=False)
    alias = torch.nn.Parameter(checkpoint, requires_grad=False)
    assert first is not alias and first.data_ptr() == alias.data_ptr()
    high, low, has_low = _split_weights(first)
    shared = _split_weights(alias)
    assert shared[0] is high and shared[1] is low and shared[2] == has_low
    # A transposed interpretation of the same pointer is distinct preparation data.
    # Use a square projection slice so both legal helper inputs keep 24 mixes.
    square = torch.randn(24, 24)
    square_first, square_transposed = _split_weights(square), _split_weights(square.t())
    assert square_first[0] is not square_transposed[0]
    key = (first.device, first.data_ptr(), tuple(first.shape), tuple(first.stride()), first.dtype)
    owner = weakref.ref(first)
    del first
    gc.collect()
    assert owner() is None and key not in _SPLIT_WEIGHTS
    # The still-live alias can repopulate the entry; the old owner was not pinned.
    replacement = _split_weights(alias)
    assert replacement[0] is not high
    torch.testing.assert_close(replacement[0], high, rtol=0, atol=0)
    torch.testing.assert_close(replacement[1], low, rtol=0, atol=0)
