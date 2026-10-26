"""Native rejection uses rank-zero probabilities and commits canonical state."""

import json
import os

import pytest
import torch.multiprocessing as mp


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1", reason="requires two explicitly assigned tensor-parallel GPUs"
)
def test_divergent_peer_probabilities_preserve_authoritative_commit(tmp_path):
    from tests.components.markov_authority_support import rank_main

    output = tmp_path / "authority.json"
    mp.spawn(rank_main, args=((tmp_path / "rendezvous").as_uri(), output), nprocs=2, join=True)
    records = json.loads(output.read_text())
    assert len(records) == 6
    assert all(record["replays"] == 12 and record["canonical_commit"] for record in records)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an explicitly assigned CUDA device")
def test_native_full_proposals_preserve_filtered_target_laws(tmp_path):
    from tests.components.markov_law_support import run

    records = run(tmp_path / "laws.json")
    assert len(records) == 6
    assert all(record["samples"] == 131072 for record in records)
