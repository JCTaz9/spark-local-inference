# SPDX-License-Identifier: Apache-2.0
"""Vocabulary-parallel softmax proposals and block rejection in prepared graphs.

The block-rejection equations and compact per-rank summaries follow Fraser's
BRRR sampler (src/brrr/vp_sampling.py, revision e23ec13). Logits, normalizers,
and retained proposal probabilities use FP32 throughout, including residuals.
Only fixed-size summaries cross ranks. Request storage and acceptance outputs
use Flare's existing speculative execution contract.
"""

import torch
import triton
import triton.language as tl

from ignition.preparation.scratch import bind_scratch
from ignition.preparation.triton import prepare_triton
from ignition.preparation.tensor_metadata import record_numeric_tensor
from .collectives import PreparedAllGather
from .markov_sampling import _row_live, _token_noise, _transition_bias, RowScaledFP8MarkovWeight
from .speculation import (
    SpeculativeAcceptance, TARGET_RNG_DOMAIN, _geometry, _metadata, _tensor,
)

# The PCIe gather channel transports FP32 bits in 64-column BF16 packs.
SUMMARY_WORDS = 32
_WORDS = tl.constexpr(32)
_TARGET_DOMAIN = tl.constexpr(TARGET_RNG_DOMAIN)


@triton.jit
def _scores(VocabularyOffset, Logits, Scores, Partials, Embedding, Head, HeadScale, Pending, Drafts,
            Histogram, Slots, Generations, LiveRows, Generation, Status, Params,
            TopK, Grammar, GrammarEnabled, Seeds, Offsets,
            logit_row_stride, logit_step_stride, hist_stride, mask_stride,
            LOCAL: tl.constexpr, VOCAB: tl.constexpr,
            WIDTH: tl.constexpr, DRAFTS: tl.constexpr, PREFIX: tl.constexpr, BLOCKS: tl.constexpr,
            CAPACITY: tl.constexpr, BATCH: tl.constexpr, RUNNING: tl.constexpr,
            MARKOV: tl.constexpr, RANK: tl.constexpr, FP8: tl.constexpr,
            SAMPLE: tl.constexpr, STEP: tl.constexpr, BLOCK: tl.constexpr):
    start = tl.load(VocabularyOffset)
    block, row = tl.program_id(0), tl.program_id(1)
    step = STEP
    if not SAMPLE:
        step = tl.program_id(2)
    col = block * BLOCK + tl.arange(0, BLOCK)
    token, inside = start + col, (col < LOCAL) & (start + col < VOCAB)
    score = tl.load(Logits + row * logit_row_stride + step * logit_step_stride + col,
                    inside, other=0.0).to(tl.float32)
    if MARKOV:
        if STEP == 0:
            previous = tl.load(Pending + row)
        else:
            previous = tl.load(Drafts + row * DRAFTS + STEP - 1)
        previous = tl.minimum(tl.maximum(previous, 0), VOCAB - 1)
        # Embeddings are replicated; transition-head rows follow the vocabulary shard.
        bias = _transition_bias(Embedding, Head, HeadScale, previous, col, inside,
                                RANK, 64, BLOCK, FP8)
        score += bias
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    temperature = tl.load(Params + safe * 6)
    repetition = tl.load(Params + safe * 6 + 3)
    presence = tl.load(Params + safe * 6 + 4)
    frequency = tl.load(Params + safe * 6 + 5)
    neutral = ((repetition == 1.0) & (presence.to(tl.int32, bitcast=True) == 0)
               & (frequency.to(tl.int32, bitcast=True) == 0))
    if not neutral:
        count = tl.load(Histogram + safe * hist_stride + token, inside, other=0)
        for index in tl.static_range(PREFIX):
            if index < step:
                draft = tl.load(Drafts + row * DRAFTS + index)
                count += (token == draft).to(tl.int32)
        adjusted = tl.where(score > 0, tl.div_rn(score, repetition), score * repetition)
        score = tl.where(count > 0, adjusted, score)
        score = score - (count > 0).to(tl.float32) * presence - count.to(tl.float32) * frequency
    score = tl.div_rn(score, tl.where(temperature > 0, temperature, 1.0))
    if tl.load(GrammarEnabled + safe) != 0:
        word = tl.load(Grammar + safe * mask_stride + token // 32, inside, other=0)
        score = tl.where(((word >> (token % 32)) & 1) != 0, score, -float("inf"))
    score = tl.where(inside, score, -float("inf"))
    maximum = tl.max(score, 0)
    finite = (maximum == maximum) & (tl.abs(maximum) < float("inf"))
    total = tl.sum(tl.exp(score - tl.where(finite, maximum, 0.0)), 0)
    bad = tl.sum((inside & ((score != score) | (score == float("inf")))).to(tl.int32), 0) != 0
    if block == 0:
        bad |= ~_row_live(Slots, Generations, LiveRows, Generation, Status, row,
                          CAPACITY, BATCH, RUNNING)
    if not SAMPLE:
        # Distributed filtering is full softmax or greedy. Reject unexpected
        # filter state rather than silently interpreting it as full softmax.
        k = tl.load(TopK + safe)
        bad |= ((k > 0) & (k < VOCAB)) | (tl.load(Params + safe * 6 + 1) != 1.0)
        bad |= tl.load(Params + safe * 6 + 2) != 0.0
    tl.store(Scores + (row * WIDTH + step) * LOCAL + col, score, col < LOCAL)
    noised = score
    if SAMPLE:
        noised += _token_noise(tl.load(Seeds + row), tl.load(Offsets + row) + 4 * STEP, token)
    index = tl.argmax(noised, 0, tie_break_left=True)
    summary_row = row
    if not SAMPLE:
        summary_row = row * WIDTH + step
    p = Partials + (summary_row * BLOCKS + block) * 5
    tl.store(p, maximum)
    tl.store(p + 1, total)
    tl.store(p + 2, tl.max(noised, 0))
    tl.store(p + 3, (start + block * BLOCK + index).to(tl.float32))
    tl.store(p + 4, bad.to(tl.float32))


@triton.jit
def _summarize(Partials, Summary, BLOCKS: tl.constexpr, PAD: tl.constexpr):
    row = tl.program_id(0)
    block = tl.arange(0, PAD)
    p = Partials + (row * BLOCKS + block) * 5
    maxima = tl.load(p, block < BLOCKS, other=-float("inf"))
    sums = tl.load(p + 1, block < BLOCKS, other=0.0)
    maximum = tl.max(maxima, 0)
    finite = (maximum == maximum) & (tl.abs(maximum) < float("inf"))
    top = tl.where(finite, maximum, 0.0)
    total = tl.sum(tl.where(block < BLOCKS, sums * tl.exp(maxima - top), 0.0), 0)
    values = tl.load(p + 2, block < BLOCKS, other=-float("inf"))
    win = tl.argmax(values, 0, tie_break_left=True)
    token = tl.sum(tl.where(block == win, tl.load(p + 3, block < BLOCKS, other=0.0), 0.0), 0)
    bad = tl.sum(tl.load(p + 4, block < BLOCKS, other=0.0), 0)
    words = tl.arange(0, _WORDS)
    tl.store(Summary + row * _WORDS + words, 0.0)
    tl.store(Summary + row * _WORDS, maximum)
    tl.store(Summary + row * _WORDS + 1, total)
    tl.store(Summary + row * _WORDS + 2, tl.max(values, 0))
    tl.store(Summary + row * _WORDS + 3, token)
    tl.store(Summary + row * _WORDS + 4, bad)


@triton.jit
def _normalize_shard(VocabularyOffset, Scores, Gathered, Probabilities, Valid, Tokens,
                     Slots, Generations, LiveRows, Generation, Status, Params,
                     LOCAL: tl.constexpr, VOCAB: tl.constexpr,
                     WIDTH: tl.constexpr, STEP: tl.constexpr, WORLD: tl.constexpr,
                     CAPACITY: tl.constexpr, BATCH: tl.constexpr, RUNNING: tl.constexpr,
                     SAMPLE: tl.constexpr, BLOCK: tl.constexpr, RANK_PAD: tl.constexpr):
    start = tl.load(VocabularyOffset)
    block, row = tl.program_id(0), tl.program_id(1)
    summary_row, step = row, STEP
    if not SAMPLE:
        row, step = summary_row // WIDTH, summary_row % WIDTH
    rank = tl.arange(0, RANK_PAD)
    ptr = Gathered + summary_row * WORLD * _WORDS + rank * _WORDS
    maxima = tl.load(ptr, rank < WORLD, other=-float("inf"))
    sums = tl.load(ptr + 1, rank < WORLD, other=0.0)
    maximum = tl.max(maxima, 0)
    finite = (maximum == maximum) & (tl.abs(maximum) < float("inf"))
    total = tl.sum(tl.where(rank < WORLD, sums * tl.exp(maxima - tl.where(finite, maximum, 0.0)), 0.0), 0)
    bad = tl.sum(tl.load(ptr + 4, rank < WORLD, other=0.0), 0)
    valid = (_row_live(Slots, Generations, LiveRows, Generation, Status, row,
                       CAPACITY, BATCH, RUNNING) & finite & (bad == 0)
             & (total > 0.0) & (total < float("inf")))
    wins = tl.load(ptr + 2, rank < WORLD, other=-float("inf"))
    win = tl.argmax(wins, 0, tie_break_left=True)
    chosen = tl.sum(tl.where(rank == win, tl.load(ptr + 3, rank < WORLD, other=0.0), 0.0), 0).to(tl.int64)
    col = block * BLOCK + tl.arange(0, BLOCK)
    score = tl.load(Scores + (row * WIDTH + step) * LOCAL + col, col < LOCAL, other=-float("inf"))
    q = tl.exp(score - maximum) / total
    if not SAMPLE:
        safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
        q = tl.where(tl.load(Params + safe * 6) > 0, q, (start + col == chosen).to(tl.float32))
    tl.store(Probabilities + (row * WIDTH + step) * LOCAL + col,
             tl.where(valid, q, (start + col == 0).to(tl.float32)), col < LOCAL)
    if block == 0:
        tl.store(Valid + row * WIDTH + step, valid)
        if SAMPLE:
            tl.store(Tokens + row * WIDTH + step, tl.where(valid, chosen, 0))


class PreparedVocabularySoftmax:
    """Full-softmax or greedy target distributions, retaining only local FP32 p.

    Top-k, top-p and min-p filters must be disabled in the declared serving
    policy. Unexpected device filter state invalidates the row. Penalties,
    temperature and grammar use global token IDs and the committed histogram.
    """

    def __init__(self, logits, probabilities, valid, requests, slots, generations, live_rows,
                 candidates, *, start, runtime, session, coordinator, name, workspace=None,
                 embedding=None, head=None, pending=None, seeds=None, offsets=None, tokens=None):
        if logits.ndim != 3 or logits.dtype not in (torch.bfloat16, torch.float32) or logits.stride(-1) != 1:
            raise ValueError("vocabulary logits require BF16/FP32 [batch,width,local vocabulary]")
        b, w, local = logits.shape
        sample = tokens is not None
        _geometry(b, w + 1 if sample else w)
        world = 1 if runtime is None else runtime.world_size
        if (type(start) is not int or start < 0 or start % local or start + local > local * world
                or not (local * (world - 1) < requests.vocab_size <= local * world)
                or requests.vocab_size > 2**24):
            raise ValueError("vocabulary shards require equal contiguous partitions covering exactly the vocabulary")
        device = logits.device
        _metadata(requests, slots, generations, live_rows, b, device)
        _tensor(probabilities, (b, w, local), torch.float32, device, "local probabilities")
        _tensor(valid, (b, w), torch.bool, device, "distribution validity")
        _tensor(candidates, (b, w if sample else w - 1), torch.int64, device, "candidate prefix IDs")
        self.markov = embedding is not None
        fp8 = isinstance(head, RowScaledFP8MarkovWeight)
        values = head.values if fp8 else head
        scale = head.scale if fp8 else None
        if sample:
            _tensor(tokens, (b, w), torch.int64, device, "proposal tokens")
            _tensor(seeds, (b,), torch.int64, device, "proposal seeds")
            _tensor(offsets, (b,), torch.int64, device, "proposal offsets")
            if self.markov:
                if (embedding.dtype != torch.bfloat16 or embedding.ndim != 2
                        or embedding.shape[0] != requests.vocab_size or not embedding.is_contiguous()
                        or values.shape != (local, embedding.shape[1]) or not values.is_contiguous()
                        or values.dtype != (torch.float8_e4m3fn if fp8 else torch.bfloat16)
                        or embedding.shape[1] % 64 or values.device != device or embedding.device != device):
                    raise ValueError("Markov proposals require replicated BF16 embeddings and a contiguous BF16/E4M3 head shard")
                _tensor(pending, (b,), torch.int64, device, "pending tokens")
                if fp8:
                    record_numeric_tensor(values)
                    record_numeric_tensor(scale)
        elif self.markov:
            raise ValueError("target softmax cannot attach proposal Markov weights")
        self.logits, self.probabilities, self.valid = logits, probabilities, valid
        self.state, self.slots, self.generations, self.live_rows = requests, slots, generations, live_rows
        self.candidates, self.start, self.world, self.batch, self.width, self.local = candidates, start, world, b, w, local
        # Rank-local vocabulary placement is a binding, so TP peers reuse the
        # same compiled computation while retaining their own global token IDs.
        self.vocabulary_offset = torch.tensor([start], dtype=torch.int32, device=device)
        self.embedding, self.head, self.scale, self.pending = embedding, values, scale, pending
        self.seeds, self.offsets, self.tokens, self.sample = seeds, offsets, tokens, sample
        summary_rows = b if sample else b * w
        self.scratch = bind_scratch(dict(scores=((b, w, local), torch.float32),
            partials=((summary_rows, triton.cdiv(local, 256), 5), torch.float32),
            summary=((summary_rows, SUMMARY_WORDS), torch.float32),
            gathered=((summary_rows, world * SUMMARY_WORDS), torch.float32)), device=device, workspace=workspace)
        self.gather = PreparedAllGather(self.scratch["summary"], self.scratch["gathered"], runtime=runtime,
            session=session, coordinator=coordinator, name=name + ".statistics")
        self.steps = None

    def prepare(self):
        from .sampling import RUNNING
        b, w, local, s, state = self.batch, self.width, self.local, self.scratch, self.state
        steps = []
        for step in range(w if self.sample else 1):
            summary_rows = b if self.sample else b * w
            score = prepare_triton(_scores, (self.vocabulary_offset, self.logits, s["scores"], s["partials"],
                self.embedding, self.head, self.scale, self.pending, self.candidates,
                state.histogram, self.slots, self.generations, self.live_rows, state.generation, state.status,
                state.parameters, state.top_k, state.grammar_mask, state.grammar_enabled, self.seeds, self.offsets,
                self.logits.stride(0), self.logits.stride(1), state.histogram.stride(0), state.grammar_mask.stride(0)),
                (triton.cdiv(local, 256), b, 1 if self.sample else w), LOCAL=local, VOCAB=state.vocab_size,
                WIDTH=w, DRAFTS=self.candidates.shape[1], PREFIX=step if self.sample else w - 1, BLOCKS=triton.cdiv(local, 256),
                CAPACITY=state.capacity, BATCH=b, RUNNING=RUNNING, MARKOV=self.markov,
                RANK=64 if self.head is None else self.head.shape[1], FP8=self.scale is not None,
                SAMPLE=self.sample, STEP=step, BLOCK=256, num_warps=4, enable_fp_fusion=False)
            summary = prepare_triton(_summarize, (s["partials"], s["summary"]), (summary_rows, 1, 1),
                BLOCKS=triton.cdiv(local, 256), PAD=triton.next_power_of_2(triton.cdiv(local, 256)), num_warps=4)
            normalize = prepare_triton(_normalize_shard, (self.vocabulary_offset, s["scores"], s["gathered"], self.probabilities,
                self.valid, self.tokens, self.slots, self.generations, self.live_rows, state.generation,
                state.status, state.parameters), (triton.cdiv(local, 256), summary_rows, 1), LOCAL=local,
                VOCAB=state.vocab_size, WIDTH=w, STEP=step, WORLD=self.world,
                CAPACITY=state.capacity, BATCH=b, RUNNING=RUNNING, SAMPLE=self.sample, BLOCK=256,
                RANK_PAD=triton.next_power_of_2(self.world), num_warps=4, enable_fp_fusion=False)
            steps.append((score, summary, normalize))
        self.steps = tuple(steps)

    def __call__(self):
        if self.steps is None:
            raise RuntimeError("vocabulary distributions require preparation")
        for score, summary, normalize in self.steps:
            score()
            summary()
            self.gather(self.scratch["summary"])
            normalize()
        return self.probabilities


@triton.jit
def _probability_partials(P, Q, Partials, LOCAL: tl.constexpr, WIDTH: tl.constexpr,
                           BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    block, row, step = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = block * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(P + (row * WIDTH + step) * LOCAL + col, col < LOCAL, other=0.0)
    q = tl.full((BLOCK,), 0.0, tl.float32)
    if step < WIDTH - 1:
        q = tl.load(Q + (row * (WIDTH - 1) + step) * LOCAL + col, col < LOCAL, other=0.0)
    bad = tl.sum(((p < 0) | (q < 0) | (p != p) | (q != q)
                  | (p == float("inf")) | (q == float("inf"))).to(tl.int32), 0) != 0
    ptr = Partials + ((row * WIDTH + step) * BLOCKS + block) * 3
    tl.store(ptr, tl.where(bad, float("nan"), tl.sum(p, 0)))
    tl.store(ptr + 1, tl.where(bad, float("nan"), tl.sum(q, 0)))
    tl.store(ptr + 2, bad.to(tl.float32))


@triton.jit
def _selected_probabilities(VocabularyOffset, P, Q, Drafts, Partials, Summary, LOCAL: tl.constexpr,
                            WIDTH: tl.constexpr,
                            BLOCKS: tl.constexpr, PAD: tl.constexpr):
    start = tl.load(VocabularyOffset)
    row = tl.program_id(0)
    word = tl.arange(0, _WORDS)
    tl.store(Summary + row * _WORDS + word, 0.0)
    block = tl.arange(0, PAD)
    for step in tl.static_range(WIDTH):
        part = Partials + ((row * WIDTH + step) * BLOCKS + block) * 3
        pmass = tl.sum(tl.load(part, block < BLOCKS, other=0.0), 0)
        qmass = tl.sum(tl.load(part + 1, block < BLOCKS, other=0.0), 0)
        tl.store(Summary + row * _WORDS + 16 + step, pmass)
        tl.store(Summary + row * _WORDS + 24 + step, qmass)
        if step < WIDTH - 1:
            token = tl.load(Drafts + row * (WIDTH - 1) + step) - start
            owned = (token >= 0) & (token < LOCAL)
            p = tl.load(P + (row * WIDTH + step) * LOCAL + token, owned, other=0.0)
            q = tl.load(Q + (row * (WIDTH - 1) + step) * LOCAL + token, owned, other=0.0)
            tl.store(Summary + row * _WORDS + step, p)
            tl.store(Summary + row * _WORDS + 8 + step, q)


@triton.jit
def _prefix(Selected, row, before, WIDTH: tl.constexpr, WORLD: tl.constexpr, RANK_PAD: tl.constexpr):
    ranks = tl.arange(0, RANK_PAD)
    ptr = Selected + row * WORLD * _WORDS + ranks * _WORDS
    log_prefix = tl.full((), 0.0, tl.float32)
    for index in tl.static_range(WIDTH - 1):
        if index < before:
            p = tl.sum(tl.load(ptr + index, ranks < WORLD, other=0.0), 0)
            q = tl.sum(tl.load(ptr + 8 + index, ranks < WORLD, other=0.0), 0)
            log_prefix = tl.minimum(log_prefix + tl.log(p) - tl.log(q), 0.0)
    return tl.exp(log_prefix)


@triton.jit
def _residual(VocabularyOffset, P, Q, Selected, Partials, Limits, Seeds, RngGeneration, RngOffset,
              Slots, Generations, CAPACITY: tl.constexpr, LOCAL: tl.constexpr,
              VOCAB: tl.constexpr, WIDTH: tl.constexpr,
              WORLD: tl.constexpr, RANK_PAD: tl.constexpr, BLOCKS: tl.constexpr,
              BLOCK: tl.constexpr, HAS_LIMITS: tl.constexpr):
    start = tl.load(VocabularyOffset)
    block, row, step = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = block * BLOCK + tl.arange(0, BLOCK)
    inside = (col < LOCAL) & (start + col < VOCAB)
    p = tl.load(P + (row * WIDTH + step) * LOCAL + col, inside, other=0.0)
    prefix = _prefix(Selected, row, step, WIDTH, WORLD, RANK_PAD)
    residual = p
    if step < WIDTH - 1:
        q = tl.load(Q + (row * (WIDTH - 1) + step) * LOCAL + col, inside, other=0.0)
        residual = tl.maximum(prefix * p - q, 0.0)
        if HAS_LIMITS:
            residual = tl.where(step < tl.load(Limits + row) - 1, residual, p)
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    seed = tl.load(Seeds + safe).to(tl.int64) ^ _TARGET_DOMAIN
    base = tl.where(tl.load(RngGeneration + safe) == tl.load(Generations + row), tl.load(RngOffset + safe), 0)
    # Token noise has a counter domain distinct from scalar acceptance draws.
    # Each conditional residual owns a separate logical offset in the reservation.
    value = tl.log(residual) + _token_noise(seed, base + step, start + col)
    value = tl.where(inside & (residual > 0) & (value == value), value, -float("inf"))
    selected = tl.argmax(value, 0, tie_break_left=True)
    part = Partials + ((row * WIDTH + step) * BLOCKS + block) * 3
    tl.store(part, tl.sum(residual, 0))
    tl.store(part + 1, tl.max(value, 0))
    tl.store(part + 2, (start + block * BLOCK + selected).to(tl.float32))


@triton.jit
def _summarize_residual(Partials, Summary, WIDTH: tl.constexpr, BLOCKS: tl.constexpr, PAD: tl.constexpr):
    row, step = tl.program_id(0), tl.program_id(1)
    block = tl.arange(0, PAD)
    part = Partials + ((row * WIDTH + step) * BLOCKS + block) * 3
    values = tl.load(part + 1, block < BLOCKS, other=-float("inf"))
    win = tl.argmax(values, 0, tie_break_left=True)
    token = tl.sum(tl.where(block == win, tl.load(part + 2, block < BLOCKS, other=0.0), 0.0), 0)
    ptr = Summary + (row * WIDTH + step) * _WORDS
    words = tl.arange(0, _WORDS)
    tl.store(ptr + words, 0.0)
    tl.store(ptr, tl.sum(tl.load(part, block < BLOCKS, other=0.0), 0))
    tl.store(ptr + 1, tl.max(values, 0))
    tl.store(ptr + 2, token)


@triton.jit
def _finish_block(Selected, Residual, Drafts, DraftValid, TargetValid, Limits,
                  Slots, Generations, LiveRows, Generation, Status, Seeds, RngGeneration, RngOffset,
                  Tokens, Count, Accepted, Offered, Valid, Increment, Survival,
                  WIDTH: tl.constexpr, WORLD: tl.constexpr, RANK_PAD: tl.constexpr,
                  CAPACITY: tl.constexpr, BATCH: tl.constexpr, VOCAB: tl.constexpr,
                  RUNNING: tl.constexpr, RESERVATION: tl.constexpr, HAS_LIMITS: tl.constexpr):
    row = tl.program_id(0)
    ranks = tl.arange(0, RANK_PAD)
    ptr = Selected + row * WORLD * _WORDS + ranks * _WORDS
    limit = WIDTH
    if HAS_LIMITS:
        limit = tl.load(Limits + row)
    valid = (_row_live(Slots, Generations, LiveRows, Generation, Status, row,
                      CAPACITY, BATCH, RUNNING) & (limit >= 1) & (limit <= WIDTH))
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    seed = tl.load(Seeds + safe).to(tl.int64) ^ _TARGET_DOMAIN
    base = tl.where(tl.load(RngGeneration + safe) == tl.load(Generations + row), tl.load(RngOffset + safe), 0)
    log_prefix = tl.full((), 0.0, tl.float32)
    accepted = tl.full((), 0, tl.int32)
    survival = tl.full((), 1.0, tl.float32)
    for step in tl.static_range(7):
        tl.store(Survival + row * 7 + step, -1.0)
    for step in tl.static_range(WIDTH):
        tl.store(Tokens + row * WIDTH + step, -1)
        if step < limit:
            pmass = tl.sum(tl.load(ptr + 16 + step, ranks < WORLD, other=0.0), 0)
            valid &= (tl.abs(pmass - 1.0) <= 1e-4) & (tl.load(TargetValid + row * WIDTH + step) != 0)
            if step < WIDTH - 1:
                if step < limit - 1:
                    qmass = tl.sum(tl.load(ptr + 24 + step, ranks < WORLD, other=0.0), 0)
                    token = tl.load(Drafts + row * (WIDTH - 1) + step)
                    valid &= ((tl.abs(qmass - 1.0) <= 1e-4) & (token >= 0) & (token < VOCAB)
                              & (tl.load(DraftValid + row * (WIDTH - 1) + step) != 0))
                    p = tl.sum(tl.load(ptr + step, ranks < WORLD, other=0.0), 0)
                    q = tl.sum(tl.load(ptr + 8 + step, ranks < WORLD, other=0.0), 0)
                    valid &= q > 0.0
                    log_prefix = tl.minimum(log_prefix + tl.log(p) - tl.log(q), 0.0)
                    prefix = tl.exp(log_prefix)
                    h = prefix
                    if step < limit - 2:
                        residual_ptr = Residual + ((row * WIDTH + step + 1) * WORLD + ranks) * _WORDS
                        weight = tl.sum(tl.load(residual_ptr, ranks < WORLD, other=0.0), 0)
                        denominator = weight + 1.0 - prefix
                        h = tl.where(denominator > 0.0, weight / denominator, 1.0)
                    # The scalar counter uses both halves of the logical offset.
                    draw = (base + step).to(tl.uint64)
                    bits, _, _, _ = tl.philox(seed, draw.to(tl.uint32), (draw >> 32).to(tl.uint32), 0, 0)
                    u = (bits.to(tl.float32) + 0.5) * 2.3283064365386963e-10
                    accepted = tl.where(u <= h, step + 1, accepted)
                    survival *= tl.minimum(tl.div_rn(p, q), 1.0)
                    tl.store(Survival + row * 7 + step, survival)
    result_ptr = Residual + ((row * WIDTH + accepted) * WORLD + ranks) * _WORDS
    values = tl.load(result_ptr + 1, ranks < WORLD, other=-float("inf"))
    win = tl.argmax(values, 0, tie_break_left=True)
    token = tl.sum(tl.where(ranks == win, tl.load(result_ptr + 2, ranks < WORLD, other=0.0), 0.0), 0).to(tl.int64)
    valid &= tl.max(values, 0) > -float("inf")
    for step in tl.static_range(WIDTH):
        result = token
        if step < WIDTH - 1:
            result = tl.where(step < accepted, tl.load(Drafts + row * (WIDTH - 1) + step), result)
        tl.store(Tokens + row * WIDTH + step, tl.where(valid & (step <= accepted), result, -1))
    tl.store(Valid + row, valid)
    tl.store(Offered + row, tl.minimum(tl.maximum(limit - 1, 0), WIDTH - 1))
    tl.store(Accepted + row, tl.where(valid, accepted, 0))
    tl.store(Count + row, tl.where(valid, accepted + 1, 0))
    tl.store(Increment + row, tl.where(valid, RESERVATION, 0))
    if not valid:
        for step in tl.static_range(7):
            tl.store(Survival + row * 7 + step, -1.0)


class PreparedVocabularyBlockVerification:
    """BRRR block acceptance with local FP32 proposal and target probabilities.

    For target probabilities p_i and the actual conditional proposals q_i, the
    block recurrence starts with r_0=1 and updates
    r_(i+1)=min(r_i*p_i(x_i)/q_i(x_i),1). Correction i has unnormalized mass
    d_i(x)=max(r_i*p_i(x)-q_i(x),0). The decision after draft i uses
    h_i=sum(d_(i+1))/(sum(d_(i+1))+1-r_(i+1)) when another draft follows,
    and h_i=r_(i+1) otherwise. The last successful decision selects the emitted
    prefix; its correction is sampled from d_i, or from p at the bonus position.
    This is the BRRR block rule; stopping at its first failed decision would
    implement a different algorithm. Log-domain recurrence preserves zero
    prefixes without overflow in a later p/q ratio.

    Normalized p and q remain on their vocabulary owner. Selected-token
    probabilities and residual winners cross ranks in fixed-size summaries.
    Scalar acceptance and token-wise residual draws use the request's target
    seed domain and complete 64-bit offsets; no batch coordinate enters RNG.
    The seeded trace differs from FlashInfer's serial-chain CDF sampling.
    """

    def __init__(self, draft_probs, drafts, target_probs, requests, slots, generations, live_rows,
                 *, start, runtime, session, coordinator, name, rng, draft_valid, target_valid,
                 token_limits=None, output=None, workspace=None):
        b, w, local = target_probs.shape
        _geometry(b, w)
        device = target_probs.device
        world = 1 if runtime is None else runtime.world_size
        if (type(start) is not int or start < 0 or start % local or start + local > local * world
                or not local * (world - 1) < requests.vocab_size <= local * world):
            raise ValueError("block verification requires equal contiguous vocabulary partitions")
        for value, shape, dtype, label in (
            (draft_probs, (b, w - 1, local), torch.float32, "proposal probabilities"),
            (target_probs, (b, w, local), torch.float32, "target probabilities"),
            (drafts, (b, w - 1), torch.int64, "draft tokens"),
            (draft_valid, (b, w - 1), torch.bool, "draft validity"),
            (target_valid, (b, w), torch.bool, "target validity"),
            (rng.generation, (requests.capacity,), torch.int64, "RNG generation"),
            (rng.target_offset, (requests.capacity,), torch.int64, "RNG offset"),
        ):
            _tensor(value, shape, dtype, device, label)
        _metadata(requests, slots, generations, live_rows, b, device)
        if token_limits is not None:
            _tensor(token_limits, (b,), torch.int32, device, "token limits")
        self.p, self.q, self.drafts = target_probs, draft_probs, drafts
        self.state, self.slots, self.generations, self.live_rows, self.rng = requests, slots, generations, live_rows, rng
        self.draft_valid, self.target_valid, self.limits = draft_valid, target_valid, token_limits
        self.batch, self.width, self.local, self.start, self.world = b, w, local, start, world
        # Rank-local vocabulary placement is a binding, so TP peers reuse the
        # same compiled computation while retaining their own global token IDs.
        self.vocabulary_offset = torch.tensor([start], dtype=torch.int32, device=device)
        self.output = SpeculativeAcceptance.allocate(b, w, device) if output is None else output
        self.output.validate(b, w, device)
        self.draw_reservation = 4 * ((w + 1 + 3) // 4)
        blocks = triton.cdiv(local, 256)
        self.scratch = bind_scratch(dict(partials=((b, w, blocks, 3), torch.float32),
            selected_summary=((b, SUMMARY_WORDS), torch.float32),
            selected=((b, world * SUMMARY_WORDS), torch.float32),
            residual_summary=((b * w, SUMMARY_WORDS), torch.float32),
            residual=((b * w, world * SUMMARY_WORDS), torch.float32)), device=device, workspace=workspace)
        self.selected_gather = PreparedAllGather(self.scratch["selected_summary"], self.scratch["selected"],
            runtime=runtime, session=session, coordinator=coordinator, name=name + ".selected")
        self.residual_gather = PreparedAllGather(self.scratch["residual_summary"], self.scratch["residual"],
            runtime=runtime, session=session, coordinator=coordinator, name=name + ".residual")
        self.kernels = None

    def prepare(self):
        from .sampling import RUNNING
        b, w, local, state, s, out = self.batch, self.width, self.local, self.state, self.scratch, self.output
        blocks = triton.cdiv(local, 256)
        partials = prepare_triton(_probability_partials, (self.p, self.q, s["partials"]), (blocks, b, w),
            LOCAL=local, WIDTH=w, BLOCKS=blocks, BLOCK=256, num_warps=4)
        selected = prepare_triton(_selected_probabilities,
            (self.vocabulary_offset, self.p, self.q, self.drafts, s["partials"], s["selected_summary"]), (b, 1, 1),
            LOCAL=local, WIDTH=w, BLOCKS=blocks,
            PAD=triton.next_power_of_2(blocks), num_warps=4)
        residual = prepare_triton(_residual, (self.vocabulary_offset, self.p, self.q, s["selected"], s["partials"], self.limits, state.seed,
            self.rng.generation, self.rng.target_offset, self.slots, self.generations), (blocks, b, w),
            CAPACITY=state.capacity, LOCAL=local, VOCAB=state.vocab_size,
            WIDTH=w, WORLD=self.world, RANK_PAD=triton.next_power_of_2(self.world),
            BLOCKS=blocks, BLOCK=256, HAS_LIMITS=self.limits is not None, num_warps=4, enable_fp_fusion=False)
        summary = prepare_triton(_summarize_residual, (s["partials"], s["residual_summary"]), (b, w, 1),
            WIDTH=w, BLOCKS=blocks, PAD=triton.next_power_of_2(blocks), num_warps=4)
        finish = prepare_triton(_finish_block, (s["selected"], s["residual"], self.drafts,
            self.draft_valid, self.target_valid, self.limits, self.slots, self.generations, self.live_rows,
            state.generation, state.status, state.seed, self.rng.generation, self.rng.target_offset,
            out.token_ids, out.proposed_count, out.accepted_drafts, out.offered_drafts,
            out.valid, out.rng_increment, out.survival), (b, 1, 1), WIDTH=w, WORLD=self.world,
            RANK_PAD=triton.next_power_of_2(self.world), CAPACITY=state.capacity, BATCH=b,
            VOCAB=state.vocab_size, RUNNING=RUNNING, RESERVATION=self.draw_reservation,
            HAS_LIMITS=self.limits is not None, num_warps=4, enable_fp_fusion=False)
        self.kernels = partials, selected, residual, summary, finish

    def __call__(self):
        if self.kernels is None:
            raise RuntimeError("vocabulary block verification requires preparation")
        partials, selected, residual, summary, finish = self.kernels
        partials()
        selected()
        self.selected_gather(self.scratch["selected_summary"])
        residual()
        summary()
        self.residual_gather(self.scratch["residual_summary"])
        finish()
        return self.output
