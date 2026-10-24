"""Quantized proposal heads preserve target weights and replay with bounded storage."""

import os

import pytest
import torch

from ignition.components.vocabulary import PreparedVocabularyProjection, mxfp8_draft_head, nvfp4_draft_head
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 5, 20, 40, 160])
@pytest.mark.parametrize("hidden", [5120, 6144])
@torch.no_grad()
def test_nvfp4_vocabulary_shard_keeps_target_weights(rows, hidden):
    device = torch.device("cuda:0")
    torch.manual_seed(1017)
    # A TP4 DeepSeek vocabulary shard has 64-aligned rows, requiring scale padding.
    target = torch.randn((32320, hidden), device=device, dtype=torch.bfloat16) * .02
    original = target.clone()
    weight = nvfp4_draft_head(target)
    assert torch.equal(target, original)
    source = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16) * .1
    local = torch.empty((rows, 32320), device=device, dtype=torch.bfloat16)
    logits = torch.empty_like(local)
    with RecordedPreparationSession(device=device, checkpoint="draft-vocabulary-shard",
            workload_schema="nvfp4-a16", numerical_policy="nvfp4", compile_workers=4) as session:
        operation = PreparedVocabularyProjection(source, weight, local, logits,
            runtime=None, coordinator=None, session=session, name="draft-head")
        session.tune(show_progress=False)
        compiled = compile_forward(lambda: operation(source))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for factor in (1., -.5):
            source.mul_(factor)
            expected = torch.nn.functional.linear(source.float(), original.float())
            operation(source)
            eager = logits.clone()
            logits.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(logits, eager, atol=0, rtol=0)
            assert torch.isfinite(logits).all()
            assert (logits.float() - expected).norm() / expected.norm() < .15
            assert torch.equal(target, original)
        graph.reset()


@pytest.mark.parametrize("rows", [1, 7, 8, 32, 4096])
@torch.no_grad()
def test_mxfp8_vocabulary_shard_replays_changed_inputs_without_target_mutation(rows):
    from ignition.preparation.graphs import PreparedGraphBundle

    torch.manual_seed(41017)
    target = torch.randn((192, 512), device="cuda", dtype=torch.bfloat16) * .02
    target[:32].zero_()
    original = target.clone()
    weight = mxfp8_draft_head(target, rows_per_chunk=71)
    scales = weight.weight.scale_rows.view(torch.float8_e8m0fnu).reshape(192, 16)
    restored = (weight.weight.values.float().reshape(192, 16, 32) * scales.float()[..., None]).reshape(192, 512)
    # Compute the conversion independently, including all-zero groups and the chunk tail.
    blocks = original.float().reshape(192, 16, 32)
    maxima = blocks.abs().amax(-1)
    exponents = torch.ceil(torch.log2(torch.where(maxima > 0, maxima / 448., 1.)))
    expected_scales = torch.exp2(exponents)
    expected_values = (blocks / expected_scales[..., None]).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales.float(), expected_scales, rtol=0, atol=0)
    assert torch.equal(weight.weight.values.view(torch.uint8), expected_values.reshape(192, 512).view(torch.uint8))
    assert torch.equal(target, original)
    source = torch.randn((rows, 512), device="cuda", dtype=torch.bfloat16)
    local, logits = (torch.empty((rows, 192), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    with RecordedPreparationSession(device="cuda:0", checkpoint="mxfp8-proposal-head",
            workload_schema="a16-vocabulary", numerical_policy="mxfp8-weight-bf16-source",
            autotune_effort="medium", compile_workers=4) as session:
        operation = PreparedVocabularyProjection(source, weight, local, logits,
            runtime=None, coordinator=None, session=session, name="draft-head")
        session.tune(show_progress=False)
        assert operation.projection.plan.prepared.selection.config.mode == "a16"
        compiled = compile_forward(lambda: operation(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"head": graph}, buffers={},
            resources=(operation, compiled, source, weight, local, logits, session))
        worker = bundle.start_worker()
        try:
            for factor in (1., -.5, 0.):
                source.mul_(factor)
                expected = source.double() @ restored.double().T
                logits.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for _ in range(11):
                    ticket = worker.submit(0)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                bound = expected.abs() * 2**-7 + expected.abs().max() * 1e-6
                assert bool(((logits.double() - expected).abs() <= bound).all())
                assert torch.equal(target, original)
        finally:
            worker.close()
            graph.reset()


@torch.no_grad()
def test_mxfp8_head_retains_the_fused_markov_proposal_distribution():
    from ignition.preparation.graphs import PreparedGraphBundle
    from tests.components.test_markov_sampling_cuda import fixture, philox

    torch.manual_seed(41018)
    proposal = fixture(batch=1, depth=7, vocab=1024)
    markov_head, embedding = proposal.head, proposal.embedding
    saved_markov, saved_embedding = markov_head.clone(), embedding.clone()
    target = torch.randn((1024, 512), device="cuda", dtype=torch.bfloat16) * .02
    saved_target = target.clone()
    weight = mxfp8_draft_head(target)
    source = torch.randn((7, 512), device="cuda", dtype=torch.bfloat16)
    local = torch.empty((7, 1024), device="cuda", dtype=torch.bfloat16)
    with RecordedPreparationSession(device="cuda:0", checkpoint="mxfp8-draft-markov-proposal",
            workload_schema="bf16-markov-fp32-probabilities", numerical_policy="mxfp8-draft-bf16-target",
            autotune_effort="medium", compile_workers=4) as session:
        head = PreparedVocabularyProjection(source, weight, local, proposal.base.view(7, 1024),
            runtime=None, coordinator=None, session=session, name="draft-head")
        session.tune(show_progress=False)

        def run():
            head(source)
            proposal()

        compiled = compile_forward(run)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"proposal": graph}, buffers={},
            resources=(compiled, head, proposal, source, weight, local, session))
        worker = bundle.start_worker()
        try:
            for factor in (1., -.5, 0.):
                source.mul_(factor)
                proposal.offsets.add_(28)
                proposal.base.fill_(float("nan"))
                proposal.probabilities.fill_(float("nan"))
                proposal.tokens.fill_(-1)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for _ in range(11):
                    ticket = worker.submit(0)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                assert proposal.valid.all() and torch.isfinite(proposal.probabilities).all()
                assert (proposal.probabilities.gather(2, proposal.tokens[..., None]) > 0).all()
                previous = proposal.pending
                seed, offset = int(proposal.seeds[0]), int(proposal.offsets[0])
                for step in range(7):
                    scores = proposal.base[:, step].double() + embedding[previous].double() @ markov_head.double().T
                    expected = scores.softmax(-1)
                    actual = proposal.probabilities[:, step]
                    torch.testing.assert_close(actual.double(), expected, atol=1e-10, rtol=3e-6)
                    # Independent Philox values reproduce the Gumbel draw from the retained q.
                    draw = offset + 4 * step
                    bits = torch.tensor([philox(seed, (token, draw & 0xFFFFFFFF, draw >> 32, 0x47554D42))[0]
                                         for token in range(1024)], dtype=torch.int64)
                    uniform = ((bits >> 16).float() * 65536. + (bits & 65535).float() + .5) * 2**-32
                    noise = -torch.log(-torch.log1p(-uniform))
                    selected = (actual[0].cpu().log() + noise).argmax()
                    assert int(proposal.tokens[0, step]) == int(selected)
                    previous = proposal.tokens[:, step]
                assert proposal.head is markov_head and proposal.embedding is embedding
                assert torch.equal(markov_head, saved_markov) and torch.equal(embedding, saved_embedding)
                assert torch.equal(target, saved_target)
        finally:
            worker.close()
            graph.reset()
