"""Prepared layer variants retain catalog storage and isolate candidate plans."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.deepseek41.contextual_selection import prepared_layer_variant


class Plan:
    def __init__(self, size=0):
        self.size = size
        self.selection = object()
        self.prepared = object()

    def same_declaration_except_override(self, other):
        return isinstance(other, Plan)

    def memory_requirements(self):
        return SimpleNamespace(scratch_nbytes=self.size)


def catalog_layer():
    output = torch.empty(8, 16, dtype=torch.bfloat16)
    qkv = SimpleNamespace(plan=Plan(), output=output)
    norm = SimpleNamespace(plan=Plan(), output=output)
    query = SimpleNamespace(plan=Plan(), output=output)
    chain = SimpleNamespace(encoded_projection=query, projections=(query,),
                            workspace=torch.empty(32, dtype=torch.uint8))
    attention = SimpleNamespace(projections={"qkv_a": qkv, "q_b": query},
                                norm_quant=norm, concurrent_query=chain)
    return SimpleNamespace(attention=attention, output=output)


def test_variant_rebinds_only_dispatch_owners_and_shares_catalog_storage():
    layer = catalog_layer()
    alternatives = (Plan(), Plan(), Plan(64))
    side = torch.empty(64, dtype=torch.uint8)
    variant = prepared_layer_variant(layer, qkv_plan=alternatives[0], norm_plan=alternatives[1],
                                     query_plan=alternatives[2], side_workspace=side)
    assert variant is not layer and variant.attention is not layer.attention
    assert variant.attention.projections is not layer.attention.projections
    assert variant.attention.projections["qkv_a"].plan is alternatives[0]
    assert variant.attention.norm_quant.plan is alternatives[1]
    assert variant.attention.projections["q_b"].plan is alternatives[2]
    assert variant.attention.concurrent_query.encoded_projection is variant.attention.projections["q_b"]
    assert variant.attention.concurrent_query.workspace is side
    assert variant.output is layer.output
    assert variant.attention.projections["q_b"].output is layer.attention.projections["q_b"].output
    assert layer.attention.projections["q_b"].plan is not alternatives[2]


def test_variant_rejects_unprepared_plan_and_undersized_overlap_storage():
    layer = catalog_layer()
    qkv, norm, query = Plan(), Plan(), Plan(64)
    query.prepared = None
    with pytest.raises(ValueError, match="prepared operator"):
        prepared_layer_variant(layer, qkv_plan=qkv, norm_plan=norm, query_plan=query,
                               side_workspace=torch.empty(64, dtype=torch.uint8))
    query.prepared = object()
    with pytest.raises(ValueError, match="caller-owned scratch"):
        prepared_layer_variant(layer, qkv_plan=qkv, norm_plan=norm, query_plan=query,
                               side_workspace=torch.empty(32, dtype=torch.uint8))
