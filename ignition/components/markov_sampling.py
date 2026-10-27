# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared full-vocabulary DSpark proposals with FP32 transition scores.

The sequential vocabulary-block schedule is adapted from vLLM's DS41 fused
Markov sampler (vllm/models/deepseek_v41/amd/ops/fused_markov.py at df3bd094).
Score rounding, explicit proposal probabilities and RNG domains here implement
Flare's probability-based rejection contract.

Philox counter words are token ID, logical draw low32, logical draw high32,
and a proposal-Gumbel domain; the request seed is the key. No batch coordinate
enters the noise. Four logical offsets remain reserved per draft position,
independent of vocabulary size. The proposal distribution is full softmax; target filters
remain the responsibility of stochastic verification.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ignition.preparation.scratch import bind_scratch
from ignition.preparation.triton import prepare_triton
from .draft_sampling import _selected_row
from .speculation import _geometry, _metadata, _tensor

# Counter word three separates token-wise noise from scalar proposal draws.
GUMBEL_COUNTER_DOMAIN = tl.constexpr(0x47554D42)


@triton.jit
def _proposal_random_bits(Seed, Offset, token):
    # A logical draw owns an entire token counter plane; neighboring positions
    # and rounds cannot reuse token noise by shifting a vocabulary-wide offset.
    draw = Offset.to(tl.uint64)
    bits, _, _, _ = tl.philox(
        Seed,
        token.to(tl.uint32),
        draw.to(tl.uint32),
        (draw >> 32).to(tl.uint32),
        tl.full((), GUMBEL_COUNTER_DOMAIN, tl.uint32),
    )
    return bits


@triton.jit
def _token_noise(Seed, Offset, token):
    bits = _proposal_random_bits(Seed, Offset, token)
    # The positive tail uses small u, avoiding cancellation in 1-u. Keep u below
    # one so a permitted token has finite noise even when every other token is masked.
    # The FP32 value immediately below one leaves every non-endpoint draw unchanged.
    u = ((bits >> 16).to(tl.float32) * 65536.0 + (bits & 65535).to(tl.float32) + 0.5) * 2.3283064365386963e-10
    u = tl.minimum(u, 0.9999999403953552)
    return -tl.log(-libdevice.log1p(-u))


@triton.jit
def _previous(Maximum, Arg, row, BLOCKS: tl.constexpr, PAD: tl.constexpr):
    index = tl.arange(0, PAD)
    maximum = tl.load(Maximum + row * BLOCKS + index, index < BLOCKS, other=-float("inf"))
    selected = tl.argmax(maximum, 0, tie_break_left=True)
    return tl.load(Arg + row * BLOCKS + selected)


@triton.jit
def _row_live(
    Slots,
    Generations,
    LiveRows,
    Generation,
    Status,
    row,
    CAPACITY: tl.constexpr,
    BATCH: tl.constexpr,
    RUNNING: tl.constexpr,
):
    live_rows = tl.load(LiveRows)
    selected, slot, safe = _selected_row(Slots, Generations, Generation, live_rows, row, CAPACITY, BATCH)
    sharing = 0
    for other in tl.static_range(BATCH):
        other_selected, other_slot, _ = _selected_row(Slots, Generations, Generation, live_rows, other, CAPACITY, BATCH)
        sharing += (other_selected & (slot == other_slot)).to(tl.int32)
    return selected & (sharing <= 1) & (tl.load(Status + safe) == RUNNING)


@triton.jit
def _markov_step(
    Base,
    Embed,
    Head,
    Pending,
    Tokens,
    Scores,
    LseMax,
    LseSum,
    Bad,
    PreviousMax,
    PreviousArg,
    Maximum,
    Arg,
    Seeds,
    Offsets,
    Histogram,
    Slots,
    Generations,
    LiveRows,
    Generation,
    Status,
    Params,
    Grammar,
    GrammarEnabled,
    base_row_stride,
    base_step_stride,
    hist_stride,
    mask_stride,
    V: tl.constexpr,
    RANK: tl.constexpr,
    DEPTH: tl.constexpr,
    STEP: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAD: tl.constexpr,
    CAPACITY: tl.constexpr,
    BATCH: tl.constexpr,
    RUNNING: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    block, row = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    inside = token < V
    if STEP == 0:
        previous = tl.load(Pending + row)
    else:
        previous = _previous(PreviousMax, PreviousArg, row, BLOCKS, PAD)
        prior_block = tl.arange(0, PAD)
        previous_maxima = tl.load(
            LseMax + (row * DEPTH + STEP - 1) * BLOCKS + prior_block, prior_block < BLOCKS, other=-float("inf")
        )
        previous_bad = tl.load(Bad + (row * DEPTH + STEP - 1) * BLOCKS + prior_block, prior_block < BLOCKS, other=0)
        previous_maximum = tl.max(previous_maxima, 0)
        previous_valid = (
            _row_live(Slots, Generations, LiveRows, Generation, Status, row, CAPACITY, BATCH, RUNNING)
            & (tl.sum(previous_bad.to(tl.int32), 0) == 0)
            & (previous_maximum == previous_maximum)
            & (tl.abs(previous_maximum) < float("inf"))
        )
        previous = tl.where(previous_valid, previous, 0)
        if block == 0:
            tl.store(Tokens + row * DEPTH + STEP - 1, previous)
    previous = tl.minimum(tl.maximum(previous, 0), V - 1)
    bias = tl.full((BLOCK,), 0.0, tl.float32)
    for start in tl.static_range(0, RANK, CHUNK):
        column = start + tl.arange(0, CHUNK)
        weights = tl.load(Head + token[:, None].to(tl.int64) * RANK + column[None, :], inside[:, None], other=0.0).to(
            tl.float32
        )
        embedding = tl.load(Embed + previous.to(tl.int64) * RANK + column).to(tl.float32)
        bias += tl.sum(weights * embedding[None, :], 1)
    score = (
        tl.load(Base + row * base_row_stride + STEP * base_step_stride + token, inside, other=0.0).to(tl.float32) + bias
    )
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    temperature = tl.load(Params + safe * 6)
    repetition = tl.load(Params + safe * 6 + 3)
    presence = tl.load(Params + safe * 6 + 4)
    frequency = tl.load(Params + safe * 6 + 5)
    neutral = (
        (repetition == 1.0) & (presence.to(tl.int32, bitcast=True) == 0) & (frequency.to(tl.int32, bitcast=True) == 0)
    )
    if not neutral:
        count = tl.load(Histogram + safe * hist_stride + token, inside, other=0)
        for index in tl.static_range(STEP):
            if index == STEP - 1:
                candidate = previous
            else:
                candidate = tl.load(Tokens + row * DEPTH + index)
            count += (token == candidate).to(tl.int32)
        adjusted = tl.where(score > 0, tl.div_rn(score, repetition), score * repetition)
        score = tl.where(count > 0, adjusted, score)
        score = score - (count > 0).to(tl.float32) * presence - count.to(tl.float32) * frequency
    score = tl.div_rn(score, tl.where(temperature > 0, temperature, 1.0))
    if tl.load(GrammarEnabled + safe) != 0:
        word = tl.load(Grammar + safe * mask_stride + token // 32, inside, other=0)
        score = tl.where(((word >> (token % 32)) & 1) != 0, score, -float("inf"))
    score = tl.where(inside, score, -float("inf"))
    score_max = tl.max(score, 0)
    finite_max = (score_max == score_max) & (tl.abs(score_max) < float("inf"))
    exponential = tl.exp(score - tl.where(finite_max, score_max, 0.0))
    partial = (row * DEPTH + STEP) * BLOCKS + block
    tl.store(LseMax + partial, score_max)
    tl.store(LseSum + partial, tl.sum(exponential, 0))
    tl.store(Bad + partial, tl.sum((inside & ((score != score) | (score == float("inf")))).to(tl.int32), 0) != 0)
    tl.store(Scores + (row * DEPTH + STEP) * V + token, score, inside)
    draw = tl.load(Offsets + row) + 4 * STEP
    noised = score + _token_noise(tl.load(Seeds + row), draw, token)
    noised = tl.where(inside & (noised == noised), noised, -float("inf"))
    index = tl.argmax(noised, 0, tie_break_left=True)
    tl.store(Maximum + row * BLOCKS + block, tl.max(noised, 0))
    tl.store(Arg + row * BLOCKS + block, block * BLOCK + index)


@triton.jit
def _normalize(
    Scores,
    LseMax,
    LseSum,
    Bad,
    Maximum,
    Arg,
    Tokens,
    Valid,
    Slots,
    Generations,
    LiveRows,
    Generation,
    Status,
    V: tl.constexpr,
    DEPTH: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAD: tl.constexpr,
    CAPACITY: tl.constexpr,
    BATCH: tl.constexpr,
    RUNNING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, row, step = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    partial = tl.arange(0, PAD)
    maxima = tl.load(LseMax + (row * DEPTH + step) * BLOCKS + partial, partial < BLOCKS, other=-float("inf"))
    sums = tl.load(LseSum + (row * DEPTH + step) * BLOCKS + partial, partial < BLOCKS, other=0.0)
    bad = tl.load(Bad + (row * DEPTH + step) * BLOCKS + partial, partial < BLOCKS, other=0)
    maximum = tl.max(maxima, 0)
    total = tl.sum(tl.where(partial < BLOCKS, sums * tl.exp(maxima - maximum), 0.0), 0)
    valid = (
        _row_live(Slots, Generations, LiveRows, Generation, Status, row, CAPACITY, BATCH, RUNNING)
        & (tl.sum(bad.to(tl.int32), 0) == 0)
        & (maximum == maximum)
        & (tl.abs(maximum) < float("inf"))
        & (total > 0.0)
        & (total < float("inf"))
    )
    token = block * BLOCK + tl.arange(0, BLOCK)
    score = tl.load(Scores + (row * DEPTH + step) * V + token, token < V, other=-float("inf"))
    q = tl.exp(score - maximum) / total
    tl.store(Scores + (row * DEPTH + step) * V + token, tl.where(valid, q, (token == 0).to(tl.float32)), token < V)
    if block == 0:
        tl.store(Valid + row * DEPTH + step, valid)
        if step == DEPTH - 1:
            tl.store(Tokens + row * DEPTH + step, tl.where(valid, _previous(Maximum, Arg, row, BLOCKS, PAD), 0))


class PreparedMarkovProposals:
    """Generate a conditional proposal block and retain full FP32 rejection probabilities.

    Base logits retain BF16/FP32 storage and transition weights remain BF16.
    Transition arithmetic, logits addition and retained proposal probabilities
    are FP32. The proposal softmax intentionally omits target top-k/top-p/min-p
    filters; verification applies the actual target policy and its positive
    residual distribution.
    """

    @staticmethod
    def scratch_layout(batch, depth, vocab, *, block=256):
        blocks = triton.cdiv(vocab, block)
        return dict(
            maxima=((2, batch, blocks), torch.float32),
            arguments=((2, batch, blocks), torch.int64),
            lse_max=((batch, depth, blocks), torch.float32),
            lse_sum=((batch, depth, blocks), torch.float32),
            bad=((batch, depth, blocks), torch.bool),
        )

    def __init__(
        self,
        base,
        embedding,
        head,
        pending,
        tokens,
        probabilities,
        valid,
        requests,
        slots,
        generations,
        live_rows,
        seeds,
        offsets,
        *,
        workspace=None,
    ):
        if base.ndim != 3:
            raise ValueError("fused Markov logits require request, draft and vocabulary dimensions")
        batch, depth, vocab = base.shape
        _geometry(batch, depth + 1)
        if (
            base.dtype not in (torch.bfloat16, torch.float32)
            or base.stride(-1) != 1
            or not 1 <= depth <= 7
            or embedding.shape != head.shape
            or head.shape[0] != vocab
            or head.shape[1] % 64
            or head.dtype != torch.bfloat16
            or embedding.dtype != torch.bfloat16
            or not head.is_contiguous()
            or not embedding.is_contiguous()
            or head.device != base.device
            or embedding.device != base.device
            or base.stride(1) < vocab
            or base.stride(0) < depth * base.stride(1)
        ):
            raise ValueError(
                "fused Markov proposals require strided BF16/FP32 logits and contiguous BF16 transition tables"
            )
        if requests.vocab_size != vocab:
            raise ValueError("fused Markov vocabulary must match request state")
        _metadata(requests, slots, generations, live_rows, batch, base.device)
        for tensor, shape, dtype, name in (
            (pending, (batch,), torch.int64, "pending"),
            (tokens, (batch, depth), torch.int64, "proposal tokens"),
            (probabilities, (batch, depth, vocab), torch.float32, "proposal probabilities"),
            (valid, (batch, depth), torch.bool, "proposal validity"),
            (seeds, (batch,), torch.int64, "proposal seeds"),
            (offsets, (batch,), torch.int64, "proposal offsets"),
        ):
            _tensor(tensor, shape, dtype, base.device, name)
        self.base, self.embedding, self.head, self.pending = (base, embedding, head, pending)
        self.tokens, self.probabilities, self.valid = tokens, probabilities, valid
        self.state, self.slots, self.generations, self.live_rows = (requests, slots, generations, live_rows)
        self.seeds, self.offsets = seeds, offsets
        self.batch, self.depth, self.vocab, self.block = batch, depth, vocab, 256
        self.scratch = bind_scratch(self.scratch_layout(batch, depth, vocab), device=base.device, workspace=workspace)
        self.kernels = None

    def prepare(self):
        from .sampling import RUNNING

        b, d, v, block = self.batch, self.depth, self.vocab, self.block
        n = triton.cdiv(v, block)
        common = dict(
            V=v,
            RANK=self.head.shape[1],
            DEPTH=d,
            BLOCKS=n,
            PAD=triton.next_power_of_2(n),
            CAPACITY=self.state.capacity,
            BATCH=b,
            RUNNING=RUNNING,
            BLOCK=block,
            CHUNK=64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        kernels = []
        s = self.scratch
        for step in range(d):
            before, after = (step - 1) % 2, step % 2
            kernels.append(
                prepare_triton(
                    _markov_step,
                    (
                        self.base,
                        self.embedding,
                        self.head,
                        self.pending,
                        self.tokens,
                        self.probabilities,
                        s["lse_max"],
                        s["lse_sum"],
                        s["bad"],
                        s["maxima"][before],
                        s["arguments"][before],
                        s["maxima"][after],
                        s["arguments"][after],
                        self.seeds,
                        self.offsets,
                        self.state.histogram,
                        self.slots,
                        self.generations,
                        self.live_rows,
                        self.state.generation,
                        self.state.status,
                        self.state.parameters,
                        self.state.grammar_mask,
                        self.state.grammar_enabled,
                        self.base.stride(0),
                        self.base.stride(1),
                        self.state.histogram.stride(0),
                        self.state.grammar_mask.stride(0),
                    ),
                    (n, b, 1),
                    STEP=step,
                    **common,
                )
            )
        kernels.append(
            prepare_triton(
                _normalize,
                (
                    self.probabilities,
                    s["lse_max"],
                    s["lse_sum"],
                    s["bad"],
                    s["maxima"][(d - 1) % 2],
                    s["arguments"][(d - 1) % 2],
                    self.tokens,
                    self.valid,
                    self.slots,
                    self.generations,
                    self.live_rows,
                    self.state.generation,
                    self.state.status,
                ),
                (n, b, d),
                V=v,
                DEPTH=d,
                BLOCKS=n,
                PAD=triton.next_power_of_2(n),
                CAPACITY=self.state.capacity,
                BATCH=b,
                RUNNING=RUNNING,
                BLOCK=block,
                num_warps=4,
                enable_fp_fusion=False,
            )
        )
        self.kernels = tuple(kernels)
        self()

    def __call__(self):
        if self.kernels is None:
            raise RuntimeError("fused Markov proposals require preparation before capture")
        for kernel in self.kernels:
            kernel()


class PreparedDraftTokenBroadcast:
    """Canonicalize proposal IDs and validity while retaining rank-local full q.

    Stochastic verification must broadcast all acceptance fields, including
    survival. Only root probabilities influence the committed target sample.
    """

    def __init__(self, tokens, valid, sampled, *, runtime, session, coordinator, name, buffers=None):
        from .collectives import PreparedBroadcast

        self.tokens, self.valid, self.sampled = tokens, valid, sampled
        self.positions = torch.arange(tokens.shape[1], device=tokens.device)
        self.broadcast = PreparedBroadcast(
            (tokens, valid), runtime=runtime, session=session, coordinator=coordinator, name=name, buffers=buffers
        )

    def finish(self, active, limits):
        self.broadcast()
        self.sampled.copy_(self.tokens[:, -1])
        self.valid.logical_and_(active[:, None] & (self.positions[None, :] < limits[:, None] - 1))
