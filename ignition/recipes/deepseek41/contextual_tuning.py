"""Bounded catalog layer context for prepared projection-plan races.

The fixture uses the catalog's existing prefill and packed-verification layer
objects, weights, collectives, buffers, and continuation pool. One captured
prefill layer produces a real 128-token window; another captured producer
prepares the target layer's embedding and metadata. Trial graphs execute the
complete closed target layer with state restored before every measurement.
Serving still uses the canonical forward catalog after plan selection.
"""

from dataclasses import dataclass, replace
from functools import partial

import torch
from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
from b12x.preparation import PreparedCall

@dataclass(frozen=True)
class _Case:
    seed: int
    window: torch.Tensor
    metadata: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    indices: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    residual: torch.Tensor
    pre_mix: torch.Tensor
    extras: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()


class _PrefillLayerSetup:
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self):
        decoder = self.decoder
        scratch = decoder.scratch
        if decoder.vision is not None:
            decoder.safe_tokens.copy_(decoder.metadata.token_ids.clamp(
                0, decoder.weights.embedding.shape[0] - 1))
        decoder.embedding()
        # This setup uses only text embedding IDs. Vision injection would
        # return the embedding unchanged but read the unmaterialized feature
        # arena during preparation-only compilation.
        scratch.residual.copy_(scratch.embedding[:, None, :])
        scratch.pre_mix.zero_()
        scratch.pre_mix[:, 0].fill_(1)
        decoder.window_metadata()
        decoder.layers[0]()


class _TargetLayerSeed:
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self):
        decoder = self.decoder
        scratch = decoder.scratch
        if decoder.vision is not None:
            decoder.safe_tokens.copy_(decoder.metadata.token_ids.clamp(
                0, decoder.weights.embedding.shape[0] - 1))
        decoder.embedding()
        scratch.residual.copy_(scratch.embedding[:, None, :])
        scratch.pre_mix.zero_()
        scratch.pre_mix[:, 0].fill_(1)
        decoder.window_metadata()


class _LayerTrial:
    def __init__(self, layer):
        self.layer = layer

    def __call__(self):
        self.layer()


def _catalog_forward(forwards, signatures, predicate, label):
    matched = [(name, forwards[name], signature) for name, signature in signatures.items()
               if name in forwards and predicate(signature, forwards[name])]
    if len(matched) != 1:
        raise ValueError(f"catalog requires one {label} forward, found {len(matched)}")
    return matched[0]


def _reference_quantize(source):
    blocks = source.float().view(source.shape[0], -1, 32)
    ratio = blocks.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = ratio.view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7fffff) != 0).int()
    scale = (exponent << 23).view(torch.float32)
    values = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.view_as(source), exponent.to(torch.uint8)


class CatalogLayer0Fixture:
    """Reuse a closed layer-zero target interval from one complete graph catalog.

    ``initialize`` requires two already captured setup graphs, prepared with
    the same session and collective runtime as the catalog. It saves two
    tuning inputs and one held-out input. ``prepared_call`` supplies a normal
    b12x context candidate whose reset and input production run outside the
    stream-gated layer interval. Neither operation rewinds collective state.
    """

    PREFILL_TOKENS = 128
    TARGET_TOKENS = 8

    def __init__(self, forwards, signatures, *, start=4096, state_row=1,
                 tune_seeds=(7101, 7102), holdout_seed=7103):
        if type(start) is not int or start < self.PREFILL_TOKENS or start % 128:
            raise ValueError("contextual layer requires an aligned position after a full window")
        if type(state_row) is not int or state_row < 1:
            raise ValueError("contextual layer requires a positive backed state row")
        seeds = (*tune_seeds, holdout_seed)
        if (len(tune_seeds) != 2 or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != 3):
            raise ValueError("contextual tuning and held-out inputs require three distinct seeds")
        prefill_name, prefill, prefill_signature = _catalog_forward(forwards, signatures,
            lambda signature, forward: (signature.kind == "prefill"
                and signature.batch_capacity == 1 and signature.token_capacity == self.PREFILL_TOKENS
                and signature.speculation_enabled is True and hasattr(forward, "decoder")),
            "speculation-enabled prefill window")
        target_name, packed, target_signature = _catalog_forward(forwards, signatures,
            lambda signature, forward: (signature.kind == "speculative"
                and signature.batch_capacity == 1 and signature.token_capacity == self.TARGET_TOKENS
                and signature.packed_token_capacity >= self.TARGET_TOKENS
                and signature.draft_tokens is not None and hasattr(forward, "target")),
            "packed target verification")
        target = packed.target
        prefill_decoder, target_decoder = prefill.decoder, target.decoder
        if (prefill_decoder.mode != "prefill" or target_decoder.mode != "verify"
                or prefill_decoder.pool is not target_decoder.pool
                or prefill_decoder.metadata.positions.numel() != self.PREFILL_TOKENS
                or target_decoder.metadata.positions.numel() != self.TARGET_TOKENS
                or target.batch != 1 or target.width != self.TARGET_TOKENS):
            raise ValueError("contextual catalog has different prefill or target geometry")
        for decoder in (prefill_decoder, target_decoder):
            layer = decoder.layers[0]
            if (layer.carry or layer.post is None or layer.boundary is not None
                    or layer.engram is not None or layer.attention.ratio != 0):
                raise ValueError("catalog layer zero must have complete uncompressed mHC seams")
        pool = target_decoder.pool
        if (state_row >= int(pool.memory_capacity[0].item())
                or int(pool.memory_capacity[1].item()) < 3):
            raise ValueError("contextual fixture requires one backed row and two backed pages")
        last_start = start + (len(seeds) - 1) * self.PREFILL_TOKENS
        if last_start // 128 >= target_decoder.metadata.page_table.shape[1]:
            raise ValueError("contextual position exceeds the prepared page table")
        self.prefill_decoder, self.target_decoder = prefill_decoder, target_decoder
        self.layer_index = 0
        self.prefill_name, self.target_name = prefill_name, target_name
        self.prefill_signature, self.target_signature = prefill_signature, target_signature
        self.pool, self.start, self.state_row = pool, start, state_row
        self.tune_seeds, self.holdout_seed = tuple(tune_seeds), holdout_seed
        self.prefill_setup_forward = _PrefillLayerSetup(prefill_decoder)
        self.target_seed_forward = _TargetLayerSeed(target_decoder)
        self.cases = ()
        self.active_case = None

    def setup_signatures(self):
        return {
            "context-prefill-layer0": replace(self.prefill_signature, inputs=()),
            "context-target-seed": replace(self.target_signature, inputs=()),
        }

    def restore_setup(self, name):
        """Provide valid first-case inputs before each setup compilation/capture."""
        seed = self.tune_seeds[0]
        if name == "context-prefill-layer0" and self.prefill_decoder is not None:
            self.pool.window(0)[self.state_row].zero_()
            self._set_metadata(self.prefill_decoder.metadata, seed=seed,
                rows=self.PREFILL_TOKENS, start=self.start - self.PREFILL_TOKENS)
        elif name in ("context-target-seed", "context-compressed-target-seed"):
            self._set_metadata(self.target_decoder.metadata, seed=seed + 100000,
                rows=self.TARGET_TOKENS, start=self.start)
        else:
            raise ValueError(f"unknown contextual setup forward {name!r}")

    def _token_rows(self, seed, rows, device):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weights = self.target_decoder.weights
        vocabulary = min(int(weights.config.text["vocab_size"]), weights.embedding.shape[0])
        for decoder in (self.prefill_decoder, self.target_decoder):
            if decoder is not None and decoder.vision is not None:
                vocabulary = min(vocabulary, decoder.vision.token_base)
        if vocabulary <= 3:
            raise ValueError("contextual embedding vocabulary lacks ordinary in-range IDs")
        return torch.randint(3, vocabulary, (1, rows), generator=generator,
                             dtype=torch.int64).to(device)

    def _set_metadata(self, metadata, *, seed, rows, start):
        device = metadata.token_ids.device
        pages = torch.full_like(metadata.page_table, -1)
        target_start = start if rows == self.TARGET_TOKENS else start + self.PREFILL_TOKENS
        pages[0, (target_start - self.PREFILL_TOKENS) // 128] = 1
        pages[0, target_start // 128] = 2
        tokens = self._token_rows(seed, rows, device)
        for decoder in (self.prefill_decoder, self.target_decoder):
            if (decoder is not None and metadata is decoder.metadata
                    and decoder.vision is not None
                    and not bool((tokens < decoder.vision.token_base).all())):
                raise ValueError("contextual setup requires text-only embedding IDs")
        metadata.set(torch.tensor([start], dtype=torch.int64, device=device),
            torch.tensor([rows], dtype=torch.int32, device=device),
            torch.tensor([self.state_row], dtype=torch.int64, device=device),
            pages, tokens,
            torch.ones((1,), dtype=torch.int32, device=device))

    @staticmethod
    def _metadata_snapshot(metadata):
        snapshots = []
        for name in metadata.__dataclass_fields__:
            tensor = getattr(metadata, name)
            if not isinstance(tensor, torch.Tensor):
                continue
            if (name == "token_page_table"
                    and tensor.data_ptr() == metadata.page_table.data_ptr()
                    and tensor.stride(0) == 0):
                # Batch-one token rows are an expanded view of page_table.
                # Restore that owning tensor; the view updates without an
                # overlapping write into its stride-zero dimension.
                continue
            snapshots.append((tensor, tensor.clone()))
        return tuple(snapshots)

    @staticmethod
    def _indices_snapshot(indices):
        tensors = [indices.position, indices.window_slots,
                   indices.window_indices, indices.window_lengths]
        for shared in indices.ratios.values():
            tensors.extend(value for value in vars(shared).values() if isinstance(value, torch.Tensor))
        return tuple((tensor, tensor.clone()) for tensor in tensors)

    def initialize(self, prefill_graph, target_seed_graph):
        """Build small model-produced windows before timing any candidate graph."""
        if self.cases:
            raise RuntimeError("contextual cases are already initialized")
        device = self.pool.backing.device
        window = self.pool.window(0)[self.state_row]
        cases = []
        for ordinal, seed in enumerate((*self.tune_seeds, self.holdout_seed)):
            target_start = self.start + ordinal * self.PREFILL_TOKENS
            window.zero_()
            self._set_metadata(self.prefill_decoder.metadata, seed=seed,
                rows=self.PREFILL_TOKENS, start=target_start - self.PREFILL_TOKENS)
            prefill_graph.replay()
            torch.cuda.synchronize(device)
            expected = pack_deepseek_v41_cache_reference(
                self.prefill_decoder.layers[0].scratch.attention.window_values,
                page_size=128, cache_kind="swa").reshape_as(window)
            if not torch.equal(window, expected):
                raise AssertionError("catalog prefill layer did not produce its checkpoint-weight window")
            saved_window = window.clone()
            self._set_metadata(self.target_decoder.metadata, seed=seed + 100000,
                rows=self.TARGET_TOKENS, start=target_start)
            target_seed_graph.replay()
            torch.cuda.synchronize(device)
            scratch = self.target_decoder.scratch
            cases.append(_Case(seed, saved_window,
                self._metadata_snapshot(self.target_decoder.metadata),
                self._indices_snapshot(self.target_decoder.window_metadata),
                scratch.residual.clone(), scratch.pre_mix.clone()))
        self.cases = tuple(cases)

    def trial_forward(self, layer):
        self._check_layer(layer)
        return _LayerTrial(layer)

    def _check_layer(self, layer):
        reference = self.target_decoder.layers[0]
        if (layer.carry or layer.post is None or layer.boundary is not None
                or layer.engram is not None or layer.attention.ratio != 0
                or layer.metadata is not reference.metadata
                or layer.attention.pool is not self.pool
                or layer.attention.window_metadata is not self.target_decoder.window_metadata
                or layer.attention_reduce.runtime is not reference.attention_reduce.runtime
                or layer.mlp_reduce.runtime is not reference.mlp_reduce.runtime
                or layer.attention_reduce.runtime is None
                or layer.scratch.residual.data_ptr() != reference.scratch.residual.data_ptr()):
            raise ValueError("trial must use the catalog's closed layer-zero buffers and TP runtime")

    def _poison(self, layer):
        scratch = layer.scratch
        layer.attention.window_cache.fill_(0xA5)
        scratch.attention.query.fill_(float("nan"))
        scratch.attention.output.fill_(float("nan"))
        scratch.routed.fill_(float("nan"))

    def _restore(self, case, layer):
        self.pool.window(0)[self.state_row].copy_(case.window)
        for target, saved in (*case.metadata, *case.indices, *case.extras):
            target.copy_(saved)
        layer.scratch.residual.copy_(case.residual)
        layer.scratch.pre_mix.copy_(case.pre_mix)
        self.active_case = case

    def validate_result(self, layer):
        case = self.active_case
        if case is None:
            raise RuntimeError("contextual layer has no active input case")
        scratch = layer.scratch
        if (not bool(torch.isfinite(scratch.residual).all())
                or not bool(torch.isfinite(scratch.attention.output).all())
                or not torch.equal(self.pool.window(0)[self.state_row], case.window)):
            raise AssertionError("contextual layer changed persistent history or produced nonfinite values")
        ids = scratch.router.ids
        if not bool(((ids >= 0) & (ids < layer.router.shape[1])).all()):
            raise AssertionError("contextual layer produced an invalid routed-expert ID")

    def validate_numeric(self, layer):
        """Check actual QKV and BF16-seam encoding outside the timed race."""
        from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch

        self._check_layer(layer)
        scratch = layer.scratch
        source = scratch.attention_hc.normalized
        projection = layer.attention.projections["qkv_a"]
        encoded = projection.encoded
        if encoded is None or projection.plan.query.allow_bf16_atomic_reduction:
            raise ValueError("contextual QKV requires prepared FP32-reduced MXFP8 rows")
        expected_values, expected_scales = _reference_quantize(source)
        if (not torch.equal(encoded.values.view(torch.uint8), expected_values.view(torch.uint8))
                or not torch.equal(encoded.scale_rows.view(torch.uint8),
                                   expected_scales.unsqueeze(0))):
            raise AssertionError("QKV activation differs from the declared BF16-to-MXFP8 encoding")
        lhs = dequantize_mxfp8_rows_torch(encoded.values, encoded.scale_rows).double()
        packed = projection.weight.weight
        rhs = dequantize_mxfp8_rows_torch(packed.values, packed.scale_rows).double()
        exact = lhs @ rhs.T
        magnitude = lhs.abs() @ rhs.abs().T
        slices = projection.plan.selection.config.split_k_slices or 1
        operations = source.shape[1] + slices
        u = operations * 2.0 ** -24
        error = (u / (1.0 - u)) * magnitude
        actual = projection.output
        below = torch.nextafter(actual, torch.full_like(actual, -float("inf"))).double()
        above = torch.nextafter(actual, torch.full_like(actual, float("inf"))).double()
        error += .5 * torch.maximum(actual.double() - below,
                                     above - actual.double())
        deviation = (actual.double() - exact).abs()
        if not bool(torch.isfinite(actual).all()) or not bool((deviation <= error).all()):
            raise AssertionError("QKV exceeds FP32 accumulation and BF16 output rounding bounds")
        q = projection.output[:, :1280].double()
        weight = layer.attention.weights["q_norm"].double()
        expected_query = (q * torch.rsqrt(q.square().mean(-1, keepdim=True)
                            + layer.attention.eps) * weight).bfloat16()
        actual_query = scratch.attention.query_low
        torch.testing.assert_close(actual_query, expected_query, rtol=2**-7, atol=0)
        norm = layer.attention.norm_quant
        query_rows = (norm.encoded if norm is not None
                      else layer.attention.projections["q_b"].encoded)
        if query_rows is None:
            raise ValueError("contextual query expansion lacks prepared MXFP8 rows")
        expected_values, expected_scales = _reference_quantize(actual_query)
        if (not torch.equal(query_rows.values.view(torch.uint8),
                            expected_values.view(torch.uint8))
                or not torch.equal(query_rows.scale_rows.view(torch.uint8),
                                   expected_scales.unsqueeze(0))):
            raise AssertionError("query expansion differs from BF16-seam MXFP8 encoding")
        return dict(qkv_max_abs_vs_fp64=float(deviation.max()),
                    query_max_abs_vs_fp64=float((actual_query.float()
                        - expected_query.float()).abs().max()))

    def restore_holdout(self, layer):
        """Prepare the unused third input for a winner-only correctness replay."""
        if len(self.cases) != 3:
            raise RuntimeError("held-out contextual input is not initialized")
        self._check_layer(layer)
        self._poison(layer)
        self._restore(self.cases[2], layer)

    def restore_tune(self, layer, case_index=0):
        """Restore one tuning input before AOT binding, warmup, or trial replay."""
        if type(case_index) is not int or not 0 <= case_index < len(self.tune_seeds):
            raise ValueError("contextual tuning case index is outside the prepared inputs")
        if len(self.cases) != len(self.tune_seeds) + 1:
            raise RuntimeError("contextual tuning inputs are not initialized")
        self._check_layer(layer)
        self._poison(layer)
        self._restore(self.cases[case_index], layer)

    def validate_trial(self, layer):
        """Qualify one compiled candidate before its latency enters a race."""
        self.validate_result(layer)
        return self.validate_numeric(layer)

    def prepared_call(self, name, graph, compiled, layer):
        """Bind one captured trial to b12x's ordinary gated context race."""
        if not self.cases or not isinstance(name, str) or not name:
            raise ValueError("contextual trial requires initialized cases and a name")
        self._check_layer(layer)
        producers = tuple(partial(self._restore, case, layer)
                          for case in self.cases[:len(self.tune_seeds)])
        return PreparedCall(run=graph.replay, output=layer.scratch.residual,
            reset=partial(self._poison, layer), produce=producers[0],
            benchmark_producers=producers, validate=partial(self.validate_trial, layer),
            owners=(self, graph, compiled, layer), close=graph.reset,
            capture_safe=True)


class CatalogCompressedLayerFixture(CatalogLayer0Fixture):
    """Use a catalog layer whose compressed-attention mHC seams are closed.

    The checkpoint's closed compressed target layer reads persistent window,
    latent, and index-key pages produced by earlier layers. This bounded
    fixture fills those pages with valid packed values and a complete candidate
    prefix through the chosen context position. Its BF16 residual is a prepared
    checkpoint embedding, so native full-model requests remain the held-out
    performance authority for the actual hidden-state distribution.
    """

    def __init__(self, forwards, signatures, *, start=4096, state_row=1,
                 tune_seeds=(8101, 8102), holdout_seed=8103):
        if type(start) is not int or start < 128 or start % 128:
            raise ValueError("compressed layer context requires an aligned position")
        if type(state_row) is not int or state_row < 1:
            raise ValueError("compressed layer context requires a positive backed state row")
        seeds = (*tune_seeds, holdout_seed)
        if (len(tune_seeds) != 2 or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != 3):
            raise ValueError("compressed tuning and held-out inputs require three distinct seeds")
        target_name, packed, target_signature = _catalog_forward(forwards, signatures,
            lambda signature, forward: (signature.kind == "speculative"
                and signature.batch_capacity == 1 and signature.token_capacity == self.TARGET_TOKENS
                and signature.packed_token_capacity >= self.TARGET_TOKENS
                and signature.draft_tokens is not None and hasattr(forward, "target")),
            "packed target verification")
        target = packed.target
        decoder = target.decoder
        if (decoder.mode != "verify" or decoder.metadata.positions.numel() != self.TARGET_TOKENS
                or target.batch != 1 or target.width != self.TARGET_TOKENS):
            raise ValueError("compressed target has different packed verification geometry")
        eligible = [(index, layer) for index, layer in enumerate(decoder.layers)
                    if not layer.carry and layer.post is not None
                    and layer.engram is None and layer.boundary is None
                    and layer.attention.ratio > 0
                    and layer.attention.global_kv is None]
        if not eligible:
            raise ValueError("catalog lacks a closed compressed consumer layer")
        index, layer = eligible[0]
        pool = decoder.pool
        backed_rows = int(pool.memory_capacity[0].item())
        backed_pages = int(pool.memory_capacity[1].item())
        required_pages = (start + self.TARGET_TOKENS + 127) // 128
        if (state_row >= backed_rows or required_pages > backed_pages
                or start // 128 >= decoder.metadata.page_table.shape[1]):
            raise ValueError("compressed context exceeds backed row, page, or table bounds")
        if layer.scratch.attention.candidates.shape[1] < start + self.TARGET_TOKENS:
            raise ValueError("compressed context exceeds prepared candidate width")
        index_source = decoder.weights.config.index_source(index)
        if (index_source is None or index_source >= len(decoder.layers)
                or not decoder.layers[index_source].attention.selectors
                or layer.scratch.attention.selected.shape[1] != 512):
            raise ValueError("compressed context lacks the prepared source indexer and selected rows")
        self.prefill_decoder = None
        self.target_decoder = decoder
        self.target_name, self.target_signature = target_name, target_signature
        self.layer_index, self.source_layer, self.index_source = (
            index, layer.attention.source.layer, index_source)
        self.page_size, self.required_pages = 128 // layer.attention.ratio, required_pages
        self.pool, self.start, self.state_row = pool, start, state_row
        self.tune_seeds, self.holdout_seed = tuple(tune_seeds), holdout_seed
        self.target_seed_forward = _TargetLayerSeed(decoder)
        self.cases = ()
        self.active_case = None

    def setup_signatures(self):
        return {"context-compressed-target-seed": replace(self.target_signature, inputs=())}

    def _set_metadata(self, metadata, *, seed, rows, start):
        device = metadata.token_ids.device
        pages = torch.full_like(metadata.page_table, -1)
        count = self.required_pages
        pages[0, :count] = torch.arange(count, dtype=pages.dtype, device=device)
        metadata.set(torch.tensor([start], dtype=torch.int64, device=device),
            torch.tensor([rows], dtype=torch.int32, device=device),
            torch.tensor([self.state_row], dtype=torch.int64, device=device),
            pages, self._token_rows(seed, rows, device),
            torch.ones((1,), dtype=torch.int32, device=device))

    def initialize(self, target_seed_graph):
        """Populate every page and candidate row the compressed layer may read."""
        if self.cases:
            raise RuntimeError("compressed contextual cases are already initialized")
        from b12x.attention import dsa_indexer as dsa

        decoder = self.target_decoder
        layer = decoder.layers[self.layer_index]
        scratch = layer.scratch.attention
        indexer = decoder.layers[self.index_source].attention.selectors[0]
        device = self.pool.backing.device
        window = self.pool.window(self.layer_index)[self.state_row]
        latent = self.pool.pages[f"layer{self.source_layer}.latent_page"][:self.required_pages]
        index_keys = indexer.cache[:self.required_pages]
        count = self.start + self.TARGET_TOKENS
        cases = []
        for seed in (*self.tune_seeds, self.holdout_seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            history = (torch.randn((128, 512), generator=generator) * .125).bfloat16().to(device)
            window.copy_(pack_deepseek_v41_cache_reference(history,
                page_size=128, cache_kind="swa").reshape_as(window))
            values = (torch.randn((self.required_pages * self.page_size, 512),
                generator=generator) * .125).bfloat16().to(device)
            latent.copy_(pack_deepseek_v41_cache_reference(values,
                page_size=self.page_size, cache_kind="indexed").reshape_as(latent))
            keys = (torch.randn((self.required_pages * self.page_size, 128),
                generator=generator) * .125).bfloat16().to(device)
            slots = torch.arange(keys.shape[0], device=device, dtype=torch.int64)
            # Use the declared key writer so cache bytes match the indexer's
            # quantization plan and physical layout.
            for first in range(0, keys.shape[0], self.TARGET_TOKENS):
                last = min(first + self.TARGET_TOKENS, keys.shape[0])
                dsa.quantize_write_index_k_mxfp4(indexer.plan, keys[first:last],
                    index_k_cache=indexer.cache, slot_mapping=slots[first:last])
            scratch.candidates.fill_(-1)
            scratch.candidates[:, :count] = torch.arange(count, device=device, dtype=torch.int32)
            scratch.candidate_counts.fill_(count)
            # The consumer receives 512 in-range selected positions from its
            # preceding source indexer. A stable seeded score order supplies
            # a valid selection without executing the open-seam source layer.
            scores = torch.randn((self.TARGET_TOKENS, count), generator=generator)
            scratch.selected.copy_(scores.topk(512, dim=1).indices.to(
                device=device, dtype=torch.int32))
            self._set_metadata(decoder.metadata, seed=seed + 100000,
                rows=self.TARGET_TOKENS, start=self.start)
            target_seed_graph.replay()
            torch.cuda.synchronize(device)
            cases.append(_Case(seed, window.clone(),
                self._metadata_snapshot(decoder.metadata),
                self._indices_snapshot(decoder.window_metadata),
                layer.scratch.residual.clone(), layer.scratch.pre_mix.clone(),
                extras=tuple((tensor, tensor.clone()) for tensor in
                    (latent, index_keys, scratch.candidates,
                     scratch.candidate_counts, scratch.selected))))
        self.cases = tuple(cases)

    def _check_layer(self, layer):
        reference = self.target_decoder.layers[self.layer_index]
        if (layer.carry or layer.post is None or layer.engram is not None
                or layer.boundary is not None or layer.attention.ratio <= 0
                or layer.attention.source.layer != self.source_layer
                or layer.metadata is not reference.metadata
                or layer.attention.pool is not self.pool
                or layer.attention.window_metadata is not self.target_decoder.window_metadata
                or layer.attention_reduce.runtime is not reference.attention_reduce.runtime
                or layer.mlp_reduce.runtime is not reference.mlp_reduce.runtime
                or layer.attention_reduce.runtime is None
                or layer.scratch.residual.data_ptr() != reference.scratch.residual.data_ptr()):
            raise ValueError("compressed trial must retain the catalog's closed layer and TP runtime")

    def _poison(self, layer):
        super()._poison(layer)
        layer.scratch.attention.selected.fill_(-1)

    def _restore(self, case, layer):
        self.pool.window(self.layer_index)[self.state_row].copy_(case.window)
        for target, saved in (*case.metadata, *case.indices, *case.extras):
            target.copy_(saved)
        layer.scratch.residual.copy_(case.residual)
        layer.scratch.pre_mix.copy_(case.pre_mix)
        self.active_case = case

    def validate_result(self, layer):
        case = self.active_case
        if case is None:
            raise RuntimeError("compressed contextual layer has no active input case")
        scratch = layer.scratch
        if (not bool(torch.isfinite(scratch.residual).all())
                or not bool(torch.isfinite(scratch.attention.output).all())
                or not torch.equal(self.pool.window(self.layer_index)[self.state_row], case.window)
                or any(not torch.equal(target, saved) for target, saved in case.extras)):
            raise AssertionError("compressed trial changed live history/candidates or produced nonfinite values")
        selected = scratch.attention.selected
        if not bool(((selected >= 0) & (selected < self.start + self.TARGET_TOKENS)).all()):
            raise AssertionError("compressed trial selected an index outside the valid cache prefix")
        ids = scratch.router.ids
        if not bool(((ids >= 0) & (ids < layer.router.shape[1])).all()):
            raise AssertionError("compressed trial produced an invalid routed-expert ID")
