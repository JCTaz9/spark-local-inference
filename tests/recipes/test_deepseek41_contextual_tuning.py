"""Catalog layer contexts reject open mHC seams and preserve trial ownership."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.preparation.execution import GraphSignature
from ignition.recipes.deepseek41.contextual_tuning import (
    CatalogCompressedLayerFixture, CatalogLayer0Fixture, _Case,
    _PrefillLayerSetup, _TargetLayerSeed,
)


class _Pool:
    def __init__(self):
        self.memory_capacity = torch.tensor([3, 3], dtype=torch.int64)
        self.backing = torch.zeros((3, 512), dtype=torch.uint8)
        self._window = torch.zeros((3, 128 * 528), dtype=torch.uint8)

    def window(self, layer):
        assert layer == 0
        return self._window


def _catalog():
    pool = _Pool()
    runtime = SimpleNamespace(sequence=torch.tensor([17]))

    def decoder(rows, mode):
        metadata = Metadata.allocate(rows, 1, 36, "cpu", max_query_tokens=rows)
        indices = SimpleNamespace(position=torch.zeros(rows, dtype=torch.int64),
            window_slots=torch.zeros(rows, dtype=torch.int64),
            window_indices=torch.zeros((rows, 128), dtype=torch.int32),
            window_lengths=torch.zeros(rows, dtype=torch.int32), ratios={})
        attention_scratch = SimpleNamespace(
            window_values=torch.zeros((rows, 512), dtype=torch.bfloat16),
            query=torch.zeros((rows, 16, 512), dtype=torch.bfloat16),
            output=torch.zeros((rows, 5120), dtype=torch.bfloat16))
        scratch = SimpleNamespace(embedding=torch.zeros((rows, 5120), dtype=torch.bfloat16),
            residual=torch.zeros((rows, 4, 5120), dtype=torch.bfloat16),
            pre_mix=torch.zeros((rows, 4), dtype=torch.float32),
            attention=attention_scratch,
            routed=torch.zeros((rows, 5120), dtype=torch.bfloat16),
            router=SimpleNamespace(ids=torch.zeros((rows, 6), dtype=torch.int32)))
        attention = SimpleNamespace(ratio=0, pool=pool, window_metadata=indices,
            window_cache=torch.zeros((1, (128 + rows) * 528), dtype=torch.uint8))
        layer = SimpleNamespace(carry=False, post=object(), boundary=None, engram=None,
            attention=attention, attention_reduce=SimpleNamespace(runtime=runtime),
            mlp_reduce=SimpleNamespace(runtime=runtime),
            router=SimpleNamespace(shape=(rows, 384)), metadata=metadata, scratch=scratch)
        return SimpleNamespace(mode=mode, pool=pool, metadata=metadata, scratch=scratch,
            layers=(layer,), window_metadata=indices,
            weights=SimpleNamespace(embedding=torch.empty((8, 1)),
                config=SimpleNamespace(text={"vocab_size": 8},
                    index_source=lambda index: 1)), vision=None)

    prefill, target = decoder(128, "prefill"), decoder(8, "verify")
    forwards = {"prepared-prefill": SimpleNamespace(decoder=prefill),
                "packed-target": SimpleNamespace(
                    target=SimpleNamespace(decoder=target, batch=1, width=8))}
    signatures = {"prepared-prefill": GraphSignature("prefill", 1, 128,
                      speculation_enabled=True),
                  "packed-target": GraphSignature("speculative", 1, 8,
                      speculation_enabled=True, packed_token_capacity=8,
                      draft_tokens=7)}
    return forwards, signatures, pool, runtime, target.layers[0]


def test_closed_catalog_seam_requires_backed_rows_and_real_tp_runtime():
    forwards, signatures, pool, runtime, layer = _catalog()
    fixture = CatalogLayer0Fixture(forwards, signatures)
    assert fixture.trial_forward(layer).layer is layer
    fixture.restore_setup("context-prefill-layer0")
    assert fixture.prefill_decoder.metadata.starts.item() == 3968
    fixture.restore_setup("context-target-seed")
    assert layer.metadata.starts.item() == 4096
    assert layer.metadata.positions.tolist() == list(range(4096, 4104))
    assert layer.metadata.token_page_table.stride(0) == 0
    snapshots = fixture._metadata_snapshot(layer.metadata)
    assert all(target is not layer.metadata.token_page_table for target, _ in snapshots)
    saved_page = layer.metadata.page_table.clone()
    layer.metadata.page_table.fill_(-1)
    for target, value in snapshots:
        target.copy_(value)
    assert torch.equal(layer.metadata.page_table, saved_page)
    assert torch.equal(layer.metadata.token_page_table[0], saved_page[0])
    pool.memory_capacity[0] = 1
    with pytest.raises(ValueError, match="backed"):
        CatalogLayer0Fixture(forwards, signatures)
    pool.memory_capacity[0] = 3
    layer.carry = True
    with pytest.raises(ValueError, match="complete uncompressed"):
        CatalogLayer0Fixture(forwards, signatures)
    layer.carry = False
    layer.attention_reduce.runtime = None
    with pytest.raises(ValueError, match="TP runtime"):
        fixture.trial_forward(layer)
    assert runtime.sequence.item() == 17


def test_trial_restores_only_bounded_model_state_and_keeps_collective_sequence():
    forwards, signatures, pool, runtime, layer = _catalog()
    fixture = CatalogLayer0Fixture(forwards, signatures)
    metadata = layer.metadata
    index = fixture.target_decoder.window_metadata
    cases = []
    for seed in (1, 2, 3):
        cases.append(_Case(seed,
            torch.full_like(pool.window(0)[1], seed),
            ((metadata.starts, torch.tensor([4096 + seed], dtype=torch.int64)),),
            ((index.window_lengths, torch.full_like(index.window_lengths, seed)),),
            torch.full_like(layer.scratch.residual, seed),
            torch.full_like(layer.scratch.pre_mix, seed)))
    fixture.cases = tuple(cases)

    class Graph:
        def __init__(self):
            self.replays = 0
            self.closed = False

        def replay(self):
            self.replays += 1
            layer.scratch.attention.output.zero_()
            layer.scratch.routed.zero_()

        def reset(self):
            self.closed = True

    graph = Graph()
    call = fixture.prepared_call("qkv-norm-qb", graph, object(), layer)
    assert len(call.benchmark_producers) == 2
    pool.window(0)[1].fill_(255)
    metadata.starts.zero_()
    layer.scratch.residual.zero_()
    call.reset()
    call.benchmark_producers[1]()
    call.run()
    fixture.validate_result(layer)
    assert graph.replays == 1
    assert torch.equal(pool.window(0)[1], cases[1].window)
    assert torch.equal(layer.scratch.residual, cases[1].residual)
    assert metadata.starts.item() == 4098
    assert index.window_lengths.eq(2).all()
    assert runtime.sequence.item() == 17
    fixture.restore_tune(layer, case_index=0)
    assert metadata.starts.item() == 4097
    assert runtime.sequence.item() == 17
    fixture.restore_holdout(layer)
    assert torch.equal(pool.window(0)[1], cases[2].window)
    assert runtime.sequence.item() == 17
    call.close()
    assert graph.closed


def test_text_only_setup_uses_embedding_without_unbacked_vision_features():
    forwards, signatures, _, _, _ = _catalog()
    prefill = forwards["prepared-prefill"].decoder
    prefill.vision = SimpleNamespace(token_base=6,
        inject=lambda *_: pytest.fail("text setup must not read vision features"))
    fixture = CatalogLayer0Fixture(forwards, signatures)
    assert bool((fixture._token_rows(7101, 128, "cpu") < prefill.vision.token_base).all())

    scratch = SimpleNamespace(embedding=torch.zeros((2, 1)),
        residual=torch.zeros((2, 1, 1)), pre_mix=torch.zeros((2, 1)))
    metadata = SimpleNamespace(token_ids=torch.tensor([3, 4]))
    decoder = SimpleNamespace(vision=prefill.vision, metadata=metadata,
        weights=SimpleNamespace(embedding=torch.empty((6, 1))),
        safe_tokens=torch.zeros(2, dtype=torch.int64), scratch=scratch)
    decoder.embedding = lambda: scratch.embedding.copy_(decoder.safe_tokens[:, None])
    decoder.window_metadata = lambda: None
    decoder.layers = (lambda: None,)
    _PrefillLayerSetup(decoder)()
    assert torch.equal(scratch.residual[:, 0, 0], metadata.token_ids.float())
    _TargetLayerSeed(decoder)()
    assert torch.equal(scratch.residual[:, 0, 0], metadata.token_ids.float())


def test_compressed_context_selects_closed_catalog_seam_and_backed_page_prefix():
    forwards, signatures, pool, runtime, layer = _catalog()
    decoder = forwards["packed-target"].target.decoder
    pool.memory_capacity[1] = 34
    layer.scratch.attention.candidates = torch.full((8, 16384), -1, dtype=torch.int32)
    layer.scratch.attention.candidate_counts = torch.zeros(8, dtype=torch.int32)
    layer.scratch.attention.selected = torch.full((8, 512), -1, dtype=torch.int32)

    def compressed(*, carry, post):
        attention = SimpleNamespace(ratio=1, pool=pool,
            window_metadata=decoder.window_metadata,
                window_cache=torch.zeros((1, (128 + 8) * 528), dtype=torch.uint8),
            source=SimpleNamespace(layer=0), selectors=(object(),), global_kv=None)
        return SimpleNamespace(carry=carry, post=post, boundary=None, engram=None,
            attention=attention, attention_reduce=SimpleNamespace(runtime=runtime),
            mlp_reduce=SimpleNamespace(runtime=runtime),
            router=SimpleNamespace(shape=(8, 384)),
            metadata=decoder.metadata, scratch=layer.scratch)

    decoder.layers = (layer, compressed(carry=True, post=None),
                      compressed(carry=False, post=object()),
                      compressed(carry=False, post=object()))
    fixture = CatalogCompressedLayerFixture(forwards, signatures)
    assert fixture.layer_index == 2
    assert fixture.trial_forward(decoder.layers[2]).layer is decoder.layers[2]
    assert fixture.setup_signatures()["context-compressed-target-seed"].draft_tokens == 7
    pool.memory_capacity[1] = 32
    with pytest.raises(ValueError, match="backed"):
        CatalogCompressedLayerFixture(forwards, signatures)
