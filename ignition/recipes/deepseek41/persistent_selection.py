"""Compare complete decoder compositions on the eight-row target catalog.

Every measured candidate runs one actual prepared layer with the same weights,
restored metadata, cache pages and incoming hyperconnection state. Native graph
families and other physical capacities retain their existing composition.
"""

from copy import copy
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json

import torch
from b12x.preparation import ContextCandidate, MemoryRequirements, Plan, PreparedCall

from ignition.components.native_operator import NativeConfig, native_contract
from ignition.components.persistent_decoder import (
    PersistentDecoderCall,
    PersistentDecoderWeights,
    PersistentPrologueCall,
    PersistentPrologueWeights,
    decoder_query,
)
from ignition.components.rotary import PreparedRotaryAngles
from ignition.preparation.capture import capture_forwards
from ignition.preparation.scratch import GraphBufferPool
from ignition.progress import phase
from .contextual_tuning import CatalogLayerFixture, _LayerTrial


class _LayerFixture(CatalogLayerFixture):
    """Restore the catalog's complete layer inputs for persistent composition races."""

    def __init__(self, name, signature, decoder, index):
        super().__init__(name, signature, decoder, index)
        self.tune_seeds = (8101 + index * 3, 8102 + index * 3)
        self.holdout_seed = 8103 + index * 3


def _target_context(forwards, signatures):
    for name, signature in signatures.items():
        if (signature.kind != "speculative" or signature.batch_capacity != 1
                or signature.packed_token_capacity != 8):
            continue
        forward = forwards[name]
        while hasattr(forward, "forward"):
            forward = forward.forward
        if hasattr(forward, "target"):
            return name + ".target", signature, forward.target.decoder
    raise ValueError("--persistent-decoder requires an eight-row batch-one DSpark target catalog")


@dataclass(frozen=True)
class _CompositionQuery:
    layer: int
    rows: int
    carry: bool
    closed_post: bool
    shared_modes: tuple[int, ...]
    source: str
    constituents: str
    storage_contract: str = "peer-owned-flags-epoch-scratch-v1"


def _tactics(query):
    options = [("separated", 0), ("tail", 1), ("tail", 2)]
    if query.carry and query.rows <= 8:
        options.extend((("prologue", 0), ("prologue_tail", 1), ("prologue_tail", 2)))
    return tuple(options)


_CONTRACT = native_contract(
    component_id="deepseek41.decoder_composition",
    query_type=_CompositionQuery,
    tactics=_tactics,
)


def _member_identity(layer):
    plans = [
        layer.attention.projections["qkv_a"].plan,
        layer.attention.norm_quant.plan,
        layer.attention.projections["q_b"].plan,
        layer.attention.output_projection.plan,
        layer.attention_reduce.plan,
        layer.ffn_hc.plan,
        layer.router_projection.plan,
        layer.router.plan,
        layer.experts.plan,
        layer.shared.up_projection.plan,
        layer.shared.activation_plan,
        layer.shared.down_projection.plan,
        layer.mlp_reduce.plan,
    ]
    if layer.attention_hc is not None:
        plans.append(layer.attention_hc.plan)
    if layer.post is not None:
        plans.append(layer.post.plan)
    payload = [
        dict(
            component=plan.component_id,
            query=plan.contract.encode_query(plan.query),
            config=plan.contract.config_payload(plan.selection.config).to_dict(),
            semantic=plan.contract.semantic_version,
            programs=sorted(
                (program.dialect, program.key) for program in plan.prepared.programs
            ),
        )
        for plan in plans
    ]
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=dict
        ).encode()
    ).hexdigest()


def _outputs(layer):
    scratch = layer.scratch
    return (
        scratch.mlp_output,
        scratch.ffn_hc.residual,
        scratch.ffn_hc.post,
        scratch.ffn_hc.comb,
        scratch.pre_mix,
        *((scratch.residual,) if layer.post is not None else ()),
    )


def _validate_history(fixture, layer):
    fixture.validate_result(layer)
    if any(not bool(torch.isfinite(tensor).all()) for tensor in _outputs(layer)):
        raise AssertionError(
            "persistent decoder produced nonfinite branch or hyperconnection state"
        )


@dataclass(frozen=True)
class PersistentWeightCatalog:
    """Layer-indexed weight views retained by selected persistent operations."""

    layers: tuple


class PersistentSelection:
    """Tune complete per-layer compositions after constituent plans are ready.

    One peer channel and scratch slab per rank is shared by every serial layer
    and final graph variant. Candidate graphs are destroyed after each race;
    final serving graphs are compiled and captured from the selected callables.
    """

    def __init__(
        self,
        forwards,
        signatures,
        *,
        ordinary,
        arena,
        coordinator,
        collective,
        device,
        graph_pool,
        cache_root,
        compile_decoder,
        warmup_storage,
        progress=None,
    ):
        self.forwards, self.signatures = forwards, signatures
        self.ordinary = ordinary
        self.target_context = _target_context(forwards, signatures)
        self.arena, self.coordinator, self.collective = arena, coordinator, collective
        self.device, self.graph_pool = device, graph_pool
        self.cache_root, self.compile_decoder = cache_root, compile_decoder
        self.storage, self.progress = warmup_storage, progress
        self.channel = None
        self.selected_plans = []
        self.retained = []
        self.prepared_dense_view_bytes = 0
        self.angle_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        self.angles = {}
        self.layer_weights = {}

    def finalist_policies(self, requests):
        return self.ordinary.finalist_policies(requests)

    def _exchange(self, step, payload):
        import torch.distributed as dist

        send = dict(payload, step="persistent-" + step)
        peers = [None] * len(self.coordinator.ranks)
        dist.all_gather_object(peers, send, group=self.coordinator.group)
        if any(peer.get("step") != send["step"] for peer in peers):
            raise ValueError("persistent context ranks reached different race steps")
        return tuple(peers)

    def _capture(self, session, functions, signatures, *, before_run, label):
        return capture_forwards(
            functions,
            signatures,
            session=session,
            device=self.device,
            stream=torch.cuda.Stream(device=self.device),
            pool=self.graph_pool,
            collective=self.collective,
            coordinator=self.coordinator,
            compile_decoder=self.compile_decoder,
            progress=self.progress,
            cache_root=self.cache_root,
            retain_graphs=False,
            warmup_storage=self.storage,
            before_run=before_run,
            label=label,
        )

    def _fixture(self, index):
        return _LayerFixture(*self.target_context, index)

    def _initialize(self, session, fixture):
        if not fixture.cases:
            fixture.initialize()

    def _variant(self, layer, packed, prologue_weights, angles, tactic, shared_mode):
        candidate = copy(layer)
        candidate.attention = copy(layer.attention)
        candidate.persistent_tail = candidate.persistent_prologue = (
            candidate.persistent_angles
        ) = None
        kind, workers = tactic
        if kind == "separated":
            return candidate
        scratch = layer.scratch
        inputs = [
            scratch.attention.attended,
            layer.metadata.positions,
            layer.metadata.token_ids,
            angles.values,
            scratch.attention_hc.residual,
            scratch.attention_hc.post,
            scratch.attention_hc.comb,
            scratch.attn_mix,
            scratch.mlp_output,
            scratch.ffn_hc.residual,
            scratch.ffn_hc.post,
            scratch.ffn_hc.comb,
            scratch.pre_mix,
            scratch.router_logits,
            scratch.router.ids,
            scratch.router.weights,
            scratch.routed,
            scratch.shared.output,
            scratch.shared.gate_up,
            scratch.shared.intermediate,
        ]
        if kind in ("tail", "prologue_tail"):
            candidate.persistent_tail = PersistentDecoderCall.bind(
                inputs, packed, self.channel, workers=workers, shared_a16=shared_mode
            )
        if kind in ("prologue", "prologue_tail"):
            inputs = [
                scratch.mlp_output,
                scratch.ffn_hc.residual,
                scratch.ffn_hc.post,
                scratch.ffn_hc.comb,
                scratch.pre_mix,
                layer.metadata.positions,
                angles.values,
                scratch.attention_hc.residual,
                scratch.attention_hc.normalized,
                scratch.attention.query_low,
                scratch.attention.window_values,
                scratch.attention.query,
                scratch.attention_hc.post,
                scratch.attention_hc.comb,
                scratch.attn_mix,
            ]
            candidate.persistent_prologue = PersistentPrologueCall.bind(
                inputs, prologue_weights, self.channel
            )
            candidate.persistent_angles = angles
        return candidate

    def _race(self, session, fixture, layer):
        from flashinfer.experimental.persistent_decoder import PersistentDecoderTP4

        if self.channel is None:
            self.channel = PersistentDecoderTP4(
                self.coordinator.group,
                device=self.device,
                ipc=self.storage.peer_storage,
            )
        self._initialize(session, fixture)
        packed = PersistentDecoderWeights.pack(layer.weights)
        self.prepared_dense_view_bytes += packed.dense_nbytes
        prologue_weights = (
            PersistentPrologueWeights.pack(layer.weights) if layer.carry else None
        )
        modes = [
            layer.shared.up_projection.plan.selection.config.mode,
            layer.shared.down_projection.plan.selection.config.mode,
        ]
        shared_mode = int(modes[0] == "a16") + 2 * int(modes[1] == "a16")
        if shared_mode and fixture.tokens > 8:
            raise ValueError(
                "persistent BF16 shared projections exceed their declared scratch capacity"
            )
        angle_key = (
            id(layer.attention.window_metadata.position),
            bool(layer.attention.ratio),
        )
        if angle_key not in self.angles:
            self.angles[angle_key] = PreparedRotaryAngles(
                layer.attention.window_metadata.position,
                layer.attention.rotary.frequencies[bool(layer.attention.ratio)],
                buffers=self.angle_buffers.cursor(),
            )
        angles = self.angles[angle_key]
        kernel = decoder_query(rows=fixture.tokens, shared_a16=shared_mode)
        dependencies = self._exchange(
            "composition-dependencies",
            dict(
                kernel=sha256(repr(kernel).encode()).hexdigest(),
                members=_member_identity(layer),
                shared_mode=shared_mode,
            ),
        )
        query = _CompositionQuery(
            layer.index,
            fixture.tokens,
            layer.carry,
            layer.post is not None,
            tuple(peer["shared_mode"] for peer in dependencies),
            sha256(
                json.dumps([peer["kernel"] for peer in dependencies]).encode()
            ).hexdigest(),
            sha256(
                json.dumps([peer["members"] for peer in dependencies]).encode()
            ).hexdigest(),
        )
        plans, variants = {}, {}
        for tactic in _tactics(query):
            name = (
                "separated" if tactic[0] == "separated" else f"{tactic[0]}-{tactic[1]}"
            )
            variant = self._variant(
                layer, packed, prologue_weights, angles, tactic, shared_mode
            )
            variants[name] = variant
            plans[name] = Plan(
                contract=_CONTRACT,
                query=query,
                override=NativeConfig(tactic),
                _compile_jobs=lambda config, device: (),
                _memory_requirements=lambda config, device: MemoryRequirements(),
                _materialize=lambda selection, device, variant=variant: variant,
            )
        obligations = [
            plan.request(
                name=f"persistent.layer{layer.index}.{name}",
                prepare_call=lambda state: PreparedCall(
                    run=state, output=_outputs(state), owners=(state,)
                ),
            )
            for name, plan in plans.items()
        ]
        prepared = session.prepare_context_plans(obligations)
        prepared.close()
        functions = {name: _LayerTrial(variant) for name, variant in variants.items()}
        signatures = {
            name: fixture.trial_signature for name in functions
        }
        graphs, compiled = self._capture(
            session,
            functions,
            signatures,
            before_run=lambda name: fixture.restore_tune(variants[name]),
            label=f"Capture persistent layer {layer.index} compositions",
        )
        references = {}
        reference_ids = {}
        rejected = {}
        diagnostics = {}
        try:
            for case_index, case in enumerate(fixture.cases):
                fixture._poison(variants["separated"])
                fixture._restore(case, variants["separated"])
                graphs["separated"].replay()
                torch.cuda.synchronize(self.device)
                _validate_history(fixture, variants["separated"])
                references[case.seed] = tuple(
                    value.clone() for value in _outputs(layer)
                )
                reference_ids[case.seed] = layer.scratch.router.ids.clone()

            def checked(role, function):
                failure = None
                try:
                    function()
                except (AssertionError, ValueError) as error:
                    failure = str(error)
                peers = self._exchange("validate-" + role, dict(failure=failure))
                failures = [peer["failure"] for peer in peers]
                if any(failures):
                    raise AssertionError(
                        f"{role} validation failed across TP ranks: {failures}"
                    )

            def validate(variant):
                checked("history", lambda: _validate_history(fixture, variant))
                if (
                    variant.persistent_tail is None
                    and variant.persistent_prologue is None
                ):
                    checked(
                        "separated-arithmetic",
                        lambda: fixture.validate_numeric(variant),
                    )
                else:
                    from .persistent_validation import validate as validate_operators

                    validate_operators(
                        variant,
                        self.channel,
                        expected_ids=reference_ids[fixture.active_case.seed],
                        coordinator=self.coordinator,
                        fixture=fixture,
                    )

            qualified = []
            for name, variant in variants.items():
                diagnostics[name] = []
                for case in fixture.cases[:2]:
                    fixture._poison(variant)
                    fixture._restore(case, variant)
                    graphs[name].replay()
                    torch.cuda.synchronize(self.device)
                    try:
                        validate(variant)
                    except (AssertionError, ValueError) as error:
                        rejected[name] = str(error)
                        break
                    actual = variant.scratch.mlp_output.float()
                    reference = references[case.seed][0].float()
                    diagnostics[name].append(
                        float(
                            (actual - reference).square().mean().sqrt()
                            / reference.square().mean().sqrt().clamp_min(1e-20)
                        )
                    )
                if name in rejected:
                    if name == "separated":
                        raise AssertionError(rejected[name])
                    graphs[name].reset()
                else:
                    qualified.append(name)
            candidates = []
            for name in qualified:
                variant = variants[name]
                producers = tuple(
                    partial(fixture._restore, case, variant)
                    for case in fixture.cases[:2]
                )
                candidates.append(
                    ContextCandidate(
                        name=name,
                        plans=(plans[name],),
                        call=PreparedCall(
                            run=graphs[name].replay,
                            output=_outputs(variant),
                            reset=partial(fixture._poison, variant),
                            produce=producers[0],
                            benchmark_producers=producers,
                            validate=partial(validate, variant),
                            owners=(fixture, variant, graphs[name], compiled.get(name)),
                            close=graphs[name].reset,
                            capture_safe=True,
                        ),
                    )
                )
            with phase(self.progress, f"Compare complete decoder layer {layer.index}"):
                result = session.race_contexts(
                    tuple(candidates),
                    collective_exchange=self._exchange,
                    collective_ranks=self.coordinator.ranks,
                    max_candidates=len(candidates),
                )
            winner = variants[result.winner]
            # Context races close trial graphs; a held-out input uses a fresh capture.
            check_name = f"persistent-heldout-layer{layer.index}"
            heldout, heldout_compiled = self._capture(
                session,
                {check_name: _LayerTrial(winner)},
                {check_name: fixture.trial_signature},
                before_run=lambda name: (
                    fixture._poison(winner),
                    fixture._restore(fixture.cases[2], winner),
                ),
                label=f"Validate persistent layer {layer.index} selection",
            )
            try:
                heldout[check_name].replay()
                torch.cuda.synchronize(self.device)
                validate(winner)
            finally:
                for graph in heldout.values():
                    graph.reset()
                heldout_compiled.clear()
            chosen = plans[result.winner]
            self.selected_plans.append(chosen)
            session.release_many(
                [plan for name, plan in plans.items() if name != result.winner]
            )
            if result.winner != "separated":
                layer.persistent_tail = winner.persistent_tail
                layer.persistent_prologue = winner.persistent_prologue
                layer.persistent_angles = winner.persistent_angles
                used_tail = packed if winner.persistent_tail is not None else None
                used_prologue = (
                    prologue_weights if winner.persistent_prologue is not None else None
                )
                self.retained.append(
                    (
                        used_tail,
                        used_prologue,
                        angles if used_prologue is not None else None,
                        winner,
                    )
                )
                self.layer_weights[layer.index] = (
                    used_tail,
                    used_prologue,
                )
            return dict(
                layer=layer.index,
                rows=query.rows,
                carry=query.carry,
                winner=result.winner,
                shared_activation_modes=modes,
                latencies_us=dict(result.latencies_us),
                cache_policy=result.cache_policy,
                query=_CONTRACT.encode_query(query),
                qualified_compositions=qualified,
                rejected_compositions=rejected,
                full_control_ffn_relative_rms=diagnostics,
            )
        finally:
            for graph in graphs.values():
                graph.reset()
            compiled.clear()
            fixture.cases = ()

    def __call__(self, session, requests, finalists):
        choices, ordinary_report = self.ordinary(session, requests, finalists)
        session.bind_context_plans(choices)
        session.preview_context_bindings()
        target = self.target_context[2]
        report = []
        for index, layer in enumerate(target.layers):
            if (
                layer.engram is not None
                or layer.boundary is not None
                or layer.attention.global_kv is not None
                or (index != 0 and layer.attention.ratio <= 0)
            ):
                continue
            fixture = self._fixture(index)
            report.append(self._race(session, fixture, layer))
        for fixture in self.ordinary.fixtures:
            fixture.cases = ()
            fixture.active_case = None
        if self.channel is not None and not self.retained:
            self.channel.close()
            self.channel = None
        return choices, dict(
            attention=ordinary_report,
            persistent_decoder=dict(
                physical_rows=8,
                batch_capacity=1,
                layers=report,
                selected_layers=[
                    entry["layer"] for entry in report if entry["winner"] != "separated"
                ],
                other_capacities="separated",
            ),
        )

    def weight_catalog(self, count):
        return PersistentWeightCatalog(
            tuple(
                None
                if index not in self.layer_weights
                else dict(
                    decoder=self.layer_weights[index][0],
                    prologue=self.layer_weights[index][1],
                )
                for index in range(count)
            )
        )

    def memory_report(self):
        return dict(
            prepared_dense_view_bytes=self.prepared_dense_view_bytes,
            retained_dense_view_bytes=sum(
                item[0].dense_nbytes for item in self.retained if item[0] is not None
            ),
            retained_channel_bytes=0 if self.channel is None else self.channel.nbytes,
            physical_rows=8,
            batch_capacity=1,
        )

    def close(self):
        if self.channel is not None:
            self.channel.close()
            self.channel = None
