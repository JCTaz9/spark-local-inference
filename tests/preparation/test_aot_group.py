"""Grouped forwards retain ordinary guards and storage-alias checks."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ignition.preparation.aot import (
    BindingMismatch, BoundForward, ForwardCache, StorageBackend,
    _BindingsChecked, _CHECK_BINDINGS,
)


class Artifact:
    def __init__(self):
        self._artifacts = SimpleNamespace(compiled_fn=object.__new__(StorageBackend),
            guard_manager=SimpleNamespace(check_verbose=lambda args: "incompatible owner"))
        self.checked = []

    def guard_check(self, owner):
        return owner.guard

    def prepare_f_locals(self, owner):
        return {"forward": owner}

    def __call__(self, owner):
        if not _CHECK_BINDINGS.get():
            raise AssertionError("binding validation must not execute a forward")
        self.checked.append(owner)
        if not owner.storage_ok:
            raise BindingMismatch("compiled tensor aliases differ")
        raise _BindingsChecked


def owner(*, guard=True, storage_ok=True):
    return SimpleNamespace(guard=guard, storage_ok=storage_ok)


def test_group_reuses_one_artifact_after_each_existing_guard_and_alias_check():
    artifact = Artifact()
    first, second = owner(), owner()
    template = BoundForward(artifact, first, cache_hit=False)
    cache = object.__new__(ForwardCache)
    bound = cache.bind_group(template, {"first": first, "second": second}, signature="same")
    assert tuple(bound) == ("first", "second")
    assert all(value.artifact is artifact and value.cache_hit is False for value in bound.values())
    assert artifact.checked == [first, second]


def test_group_rejects_storage_mismatch_on_every_rank_without_recompilation(monkeypatch):
    import torch.distributed as dist

    artifact = Artifact()
    first, bad = owner(), owner(storage_ok=False)
    template = BoundForward(artifact, first, cache_hit=False)
    coordinator = SimpleNamespace(ranks=(0, 1), group=object())
    def exchange(results, local, *, group):
        results[:] = [local, (local[0], local[1], {"bad": "peer guard failure"})]
    monkeypatch.setattr(dist, "all_gather_object", exchange)
    with pytest.raises(BindingMismatch, match="compiled tensor aliases differ"):
        object.__new__(ForwardCache).bind_group(template,
            {"first": first, "bad": bad}, signature="same", coordinator=coordinator)
    assert artifact.checked == [first, bad]


def test_group_rejects_different_rank_signatures(monkeypatch):
    import torch.distributed as dist

    artifact = Artifact()
    first = owner()
    template = BoundForward(artifact, first, cache_hit=False)
    coordinator = SimpleNamespace(ranks=(0, 1), group=object())
    monkeypatch.setattr(dist, "all_gather_object",
        lambda results, local, *, group: results.__setitem__(slice(None),
            [local, (("first",), "different", {})]))
    with pytest.raises(BindingMismatch, match="different grouped forward owners or signatures"):
        object.__new__(ForwardCache).bind_group(template,
            {"first": first}, signature="same", coordinator=coordinator)


def test_trial_inputs_are_restored_before_tuning_and_frozen_inductor_warmups(tmp_path):
    events, state = [], {"ready": False}

    class Compiled:
        _flare_graph = object()

    class TrialArtifact:
        _artifacts = SimpleNamespace(compiled_fn=SimpleNamespace(artifact=Compiled()))

        def save_compiled_function(self, path):
            events.append("save")

    class TrialForward:
        artifact = TrialArtifact()
        cache_hit = True

        def __call__(self):
            assert state["ready"]
            state["ready"] = False
            events.append("run")

    class Selections:
        def register(self, bound):
            events.append("register")

        def warm(self, bound, warmup, *, before_run=None):
            for _ in range(2):
                with warmup():
                    if before_run is not None:
                        before_run()
                    bound()

        def freeze(self, bound):
            events.append("freeze")

    @contextmanager
    def transient():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def restore():
        assert not state["ready"]
        state["ready"] = True
        events.append("restore")

    bound = TrialForward()
    cache = object.__new__(ForwardCache)
    assert cache._warm(bound, tmp_path / "trial.bin", name="trial", selections=Selections(),
                       warmup=transient, before_run=restore) is bound
    assert events == ["register", "enter", "restore", "run", "exit",
                      "enter", "restore", "run", "exit", "freeze", "save"]
