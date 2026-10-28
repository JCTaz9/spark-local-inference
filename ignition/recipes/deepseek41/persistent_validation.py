"""Numerical gates for persistent decoder candidates before GPU timing.

FP64 dot products bound FP32 accumulation with gamma(n)=n*u/(1-n*u),
u=2^-24. Every declared BF16 boundary adds half the local BF16 spacing;
MXFP8 outputs add half their E4M3 spacing at the actual block scale. These
are per-operator arithmetic bounds, rather than a whole-layer RMS allowance.
Routed and shared projections retain their BF16 boundaries. Ordered route
reductions and TP reductions require exact BF16 values at those boundaries.
"""

import torch
import torch.distributed as dist


def _gamma(operations):
    value = operations * 2.0**-24
    return value / (1.0 - value)


def _rounding(actual):
    below = torch.nextafter(actual, torch.full_like(actual, -float("inf"))).double()
    above = torch.nextafter(actual, torch.full_like(actual, float("inf"))).double()
    value = actual.double()
    return 0.5 * torch.maximum(value - below, above - value)


def _check(actual, expected, error, role):
    deviation = (actual.double() - expected).abs()
    if not bool(torch.isfinite(actual).all()) or not bool((deviation <= error).all()):
        raise AssertionError(
            f"{role} exceeds FP32 accumulation and declared rounding bounds; "
            f"maximum absolute deviation {float(deviation.max()):.8g}; "
            f"maximum excess {float((deviation - error).max()):.8g}; "
            f"first actual/expected/bound {actual.flatten()[int((deviation - error).argmax())].item()},"
            f"{expected.flatten()[int((deviation - error).argmax())].item()},"
            f"{error.flatten()[int((deviation - error).argmax())].item()}"
        )


def _dot(actual, left, right, *, extra=0, fixed_error=0.0, role):
    exact = left @ right.T
    error = _gamma(left.shape[1] + extra) * (left.abs() @ right.abs().T)
    error += (left.shape[1] + extra) * 2.0**-149 + fixed_error
    if actual.dtype == torch.bfloat16:
        error += _rounding(actual)
    else:
        error += _gamma(1) * exact.abs()
    _check(actual, exact, error, role)


def _quantized(source, *, min_amax=0.0, zero_scale=False):
    grouped = source.float().view(source.shape[0], -1, 32)
    amax = grouped.abs().amax(-1).clamp_min(min_amax)
    ratio = torch.where(
        amax > 0,
        amax * (1.0 / 448.0),
        torch.zeros_like(amax) if zero_scale else torch.ones_like(amax),
    )
    bits = ratio.contiguous().view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    factor = (exponent << 23).view(torch.float32)
    inverse = torch.where(
        factor > 0, torch.reciprocal(factor), torch.zeros_like(factor)
    )
    values = (
        (grouped * inverse[..., None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view_as(source)
    )
    scales = exponent.to(torch.uint8)
    decoded = (
        values.double().view(source.shape[0], -1, 32) * factor.double()[..., None]
    ).view_as(source)
    return values, scales, decoded


def _weight(packed):
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch

    result = dequantize_mxfp8_rows_torch(packed.values, packed.scale_rows).double()
    return result[:, :, 0] if result.ndim == 3 and result.shape[2] == 1 else result


def _rms(actual, source, weight, epsilon, role):
    source = source.double()
    exact = (
        source
        * torch.rsqrt(source.square().mean(-1, keepdim=True) + epsilon)
        * weight.double()
    )
    g = _gamma(source.shape[-1] + 1)
    # A bounded sum-of-squares perturbs reciprocal RMS by at most this ratio.
    relative = (1.0 - g) ** -0.5 - 1.0 + _gamma(4)
    _check(actual, exact, exact.abs() * relative + _rounding(actual), role)


def _rotated_dot(actual, left, right, angles, positions, *, extra=0, role):
    exact = left @ right.T
    error = _gamma(left.shape[1] + extra) * (left.abs() @ right.abs().T)
    # The projection rounds to BF16 before rotary multiplication.
    error += 2.0**-8 * (exact.abs() + error) + 2.0**-134
    rows = actual.shape[0]
    exact = exact.view(rows, -1, 512)
    error = error.view_as(exact)
    cosine, sine = (
        angles.double()[..., 0][:, None, :],
        angles.double()[..., 1][:, None, :],
    )
    even, odd = exact[..., -64::2], exact[..., -63::2]
    even_error, odd_error = error[..., -64::2].clone(), error[..., -63::2].clone()
    rotated = exact.clone()
    rotated[..., -64::2] = even * cosine - odd * sine
    rotated[..., -63::2] = odd * cosine + even * sine
    error[..., -64::2] = (
        even_error * cosine.abs()
        + odd_error * sine.abs()
        + _gamma(3) * (even * cosine).abs()
        + _gamma(3) * (odd * sine).abs()
    )
    error[..., -63::2] = (
        odd_error * cosine.abs()
        + even_error * sine.abs()
        + _gamma(3) * (odd * cosine).abs()
        + _gamma(3) * (even * sine).abs()
    )
    inactive = positions < 0
    rotated[inactive] = 0
    error[inactive] = 0
    _check(
        actual.view_as(rotated),
        rotated,
        error + _rounding(actual.view_as(rotated)),
        role,
    )


def _mix(layer, residual, kind, post, comb, pre):
    weights = layer.weights.residual
    flat = residual.flatten(1).double()
    projected = (flat @ weights[f"hc_{kind}_fn"].double().T) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + layer.attention.eps
    )
    scale, base = (
        weights[f"hc_{kind}_scale"].double(),
        weights[f"hc_{kind}_base"].double(),
    )
    expected_pre = torch.sigmoid(projected[:, :4] * scale[0] + base[:4]) + 1e-6
    expected_post = 2 * torch.sigmoid(projected[:, 4:8] * scale[1] + base[4:8])
    expected_comb = (
        torch.softmax(
            projected[:, 8:].view(-1, 4, 4) * scale[2] + base[8:].view(4, 4), dim=-1
        )
        + 1e-6
    )
    expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        expected_comb = expected_comb / (expected_comb.sum(-1, keepdim=True) + 1e-6)
        expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    # Existing b12x lagged-mHC scalar qualification:
    # tests/experimental/b12x/norm/test_mhc_lagged.py and test_mhc_fused_tf32.py.
    # BF16 state and normalization use the tighter FP64 accumulation bounds above.
    for actual, expected in (
        (pre, expected_pre),
        (post, expected_post),
        (comb, expected_comb),
    ):
        torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=4e-5)


def _prologue(layer, channel, case):
    scratch = layer.scratch
    rows = scratch.mlp_output.shape[0]
    saved = {target.data_ptr(): value for target, value in case.inputs}
    residual = saved[scratch.residual.data_ptr()]
    pre_mix = saved[scratch.pre_mix.data_ptr()]
    if not torch.equal(scratch.attention_hc.residual, residual):
        raise AssertionError(
            "carry prologue changes the fixture's identity residual handoff"
        )
    collapse = (
        (pre_mix.double()[..., None] * residual.double()).sum(1).bfloat16()
    )
    _rms(
        scratch.attention_hc.normalized,
        collapse,
        layer.weights.residual["attn_norm.weight"],
        layer.attention.eps,
        "attention weighted RMS normalization",
    )
    _mix(
        layer,
        scratch.attention_hc.residual,
        "attn",
        scratch.attention_hc.post,
        scratch.attention_hc.comb,
        scratch.attn_mix,
    )
    values, scales, decoded = _quantized(scratch.attention_hc.normalized, min_amax=1e-4)
    actual_values = channel.scratch[25][: rows * 5120].view(rows, 5120)
    groups = torch.arange(160, device=values.device)
    offsets = (
        torch.arange(rows, device=values.device)[:, None] * 16
        + (groups // 4)[None, :] * 512
        + (groups % 4)[None, :]
    )
    actual_scales = channel.scratch[26][offsets]
    if not torch.equal(actual_values, values.view(torch.uint8)) or not torch.equal(
        actual_scales, scales
    ):
        raise AssertionError(
            "prologue QKV input differs from the declared floored MXFP8 encoding"
        )
    projected = channel.scratch[28].view(torch.float32)[: rows * 1792].view(rows, 1792)
    if not torch.equal(projected, projected.bfloat16().float()):
        raise AssertionError("prologue QKV projection omits its BF16 output boundary")
    _dot(
        projected.bfloat16(),
        decoded,
        _weight(layer.weights.attention["qkv_a"].weight),
        extra=5,
        role="prologue query/KV projection",
    )
    _rms(
        scratch.attention.query_low,
        projected[:, :1280],
        layer.weights.attention["q_norm"],
        layer.attention.eps,
        "prologue query RMS normalization",
    )
    kv = projected[:, 1280:].double()
    exact = (
        kv
        * torch.rsqrt(kv.square().mean(-1, keepdim=True) + layer.attention.eps)
        * layer.weights.attention["kv_norm"].double()
    )
    g = _gamma(513)
    error = exact.abs() * ((1.0 - g) ** -0.5 - 1.0 + _gamma(4))
    error += 2.0**-8 * (exact.abs() + error) + 2.0**-134
    angles = layer.persistent_angles.values.double()
    even, odd = exact[:, -64::2].clone(), exact[:, -63::2].clone()
    even_error, odd_error = error[:, -64::2].clone(), error[:, -63::2].clone()
    cosine, sine = angles[..., 0], angles[..., 1]
    exact[:, -64::2] = even * cosine - odd * sine
    exact[:, -63::2] = odd * cosine + even * sine
    error[:, -64::2] = (
        even_error * cosine.abs()
        + odd_error * sine.abs()
        + _gamma(3) * ((even * cosine).abs() + (odd * sine).abs())
    )
    error[:, -63::2] = (
        odd_error * cosine.abs()
        + even_error * sine.abs()
        + _gamma(3) * ((odd * cosine).abs() + (even * sine).abs())
    )
    inactive = layer.metadata.positions < 0
    exact[inactive] = 0
    error[inactive] = 0
    _check(
        scratch.attention.window_values,
        exact,
        error + _rounding(scratch.attention.window_values),
        "prologue BF16 KV normalization and rotary output",
    )
    _, _, query = _quantized(scratch.attention.query_low, min_amax=1e-4)
    _rotated_dot(
        scratch.attention.query,
        query,
        _weight(layer.weights.attention["q_b"].weight),
        layer.persistent_angles.values,
        layer.metadata.positions,
        role="prologue query projection and rotary output",
    )


def _local_inbox(channel, rows, which):
    epoch = int(channel.scratch[0].view(torch.int32).item())
    slot = ((which * 2) * 2 + (epoch & 1)) * 128 * 5120
    begin = 8192 + slot * 2
    return (
        channel.peers[0][begin : begin + rows * 5120 * 2]
        .view(torch.bfloat16)
        .view(rows, 5120)
    )


def _fp8_spacing(values):
    exponent = (values.view(torch.uint8).to(torch.int32) >> 3) & 15
    return torch.exp2(torch.where(exponent == 0, -9, exponent - 10).double())


def _attention_reduction(layer, channel):
    rows = layer.scratch.mlp_output.shape[0]
    epoch = int(channel.scratch[0].view(torch.int32).item())
    begin = 8192 + (epoch & 1) * 128 * 5120 * 2
    total = torch.zeros(
        (rows, 5120), dtype=torch.float32, device=layer.scratch.mlp_output.device
    )
    for rank in range(4):
        index = 0 if rank == channel.rank else rank + 1 if rank < channel.rank else rank
        values = (
            channel.peers[index][begin : begin + rows * 5120 * 2]
            .view(torch.bfloat16)
            .view(rows, 5120)
        )
        total.add_(values.float())
    actual = channel.scratch[7][: rows * 5120 * 2].view(torch.bfloat16).view(rows, 5120)
    if not torch.equal(actual, total.bfloat16()):
        raise AssertionError(
            "attention TP reduction differs from rank-ordered FP32 sum and BF16 output"
        )


def _wo(layer, channel):
    scratch = layer.scratch
    rows = scratch.attention.attended.shape[0]
    source = scratch.attention.attended.view(rows, 8192)
    values, scales, decoded = _quantized(source)
    actual_values = channel.scratch[32][: rows * 8192].view(rows, 8192)
    actual_scales = channel.scratch[33][: rows * 256].view(rows, 256)
    if not torch.equal(actual_values, values.view(torch.uint8)) or not torch.equal(
        actual_scales, scales
    ):
        raise AssertionError(
            "WO-A input does not preserve its BF16-to-MXFP8 quantization contract"
        )
    weights = layer.weights.attention["output"]
    rhs = _weight(weights.wo_a).permute(2, 0, 1)
    lhs = decoded.view(rows, 2, 4096).permute(1, 0, 2)
    exact = lhs @ rhs.transpose(1, 2)
    magnitude = lhs.abs() @ rhs.abs().transpose(1, 2)
    exact = exact.permute(1, 0, 2).reshape(rows, 2048)
    error = (_gamma(4096 + 16) * magnitude).permute(1, 0, 2).reshape(rows, 2048)
    error += 2.0**-8 * (exact.abs() + error) + 2.0**-134
    zvalues = (
        channel.scratch[4][: rows * 2048].view(torch.float8_e4m3fn).view(rows, 2048)
    )
    zscales = channel.scratch[5][: rows * 64].view(rows, 64)
    factors = torch.exp2(zscales.double() - 127).repeat_interleave(32, dim=-1)
    quant = zvalues.double()
    # E4M3 normal spacing is 2^(floor(log2(abs(x)))-3), subnormal spacing 2^-9.
    spacing = _fp8_spacing(zvalues)
    error += 0.5 * spacing * factors
    _check(
        quant * factors, exact, error, "WO-A projection with BF16 and MXFP8 boundaries"
    )
    _dot(
        _local_inbox(channel, rows, 0),
        quant * factors,
        _weight(weights.wo_b),
        extra=4,
        role="WO-B projection",
    )


def _post_pre(layer, channel):
    scratch = layer.scratch
    rows = scratch.mlp_output.shape[0]
    attention = (
        channel.scratch[7][: rows * 5120 * 2].view(torch.bfloat16).view(rows, 5120)
    )
    residual = scratch.attention_hc.residual.double()
    post = scratch.attention_hc.post.double()
    comb = scratch.attention_hc.comb.double()
    exact = post[..., None] * attention.double()[:, None, :] + torch.einsum(
        "bij,bih->bjh", comb, residual
    )
    magnitude = post.abs()[..., None] * attention.double().abs()[
        :, None, :
    ] + torch.einsum("bij,bih->bjh", comb.abs(), residual.abs())
    _check(
        scratch.ffn_hc.residual,
        exact,
        _gamma(5) * magnitude + _rounding(scratch.ffn_hc.residual),
        "attention residual injection",
    )
    collapsed = (
        channel.scratch[10][: rows * 5120 * 2].view(torch.bfloat16).view(rows, 5120)
    )
    terms = scratch.attn_mix.double()[..., None] * scratch.ffn_hc.residual.double()
    _check(
        collapsed,
        terms.sum(1),
        _gamma(4) * terms.abs().sum(1) + _rounding(collapsed),
        "lagged hyperconnection collapse",
    )
    normalized = (
        channel.scratch[12][: rows * 5120 * 2].view(torch.bfloat16).view(rows, 5120)
    )
    _rms(
        normalized,
        collapsed,
        layer.weights.residual["ffn_norm.weight"],
        layer.attention.eps,
        "feed-forward weighted RMS normalization",
    )
    _dot(
        scratch.router_logits,
        normalized.double(),
        layer.weights.experts.router.double(),
        extra=40,
        fixed_error=40 * 0.5 * 2.0**-32,
        role="FP32 routing projection",
    )
    scores = torch.sqrt(torch.nn.functional.softplus(scratch.router_logits.float()))
    keys = scores + layer.weights.experts.correction.float()
    # Reversing expert columns before stable sort gives descending-ID ties.
    expected_ids = keys.flip(-1).argsort(dim=-1, descending=True, stable=True)[:, :6]
    expected_ids = 383 - expected_ids
    if not torch.equal(scratch.router.ids, expected_ids.to(scratch.router.ids.dtype)):
        raise AssertionError(
            "persistent routing differs from sqrt-softplus top-k with descending-ID ties"
        )
    chosen = scores.gather(1, expected_ids)
    expected_weights = chosen / chosen.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(
        scratch.router.weights, expected_weights, rtol=16 * 2.0**-24, atol=2.0**-149
    )
    _mix(
        layer,
        scratch.ffn_hc.residual,
        "ffn",
        scratch.ffn_hc.post,
        scratch.ffn_hc.comb,
        scratch.pre_mix,
    )
    return normalized


def _shared(layer, channel, normalized):
    scratch = layer.scratch
    rows = normalized.shape[0]
    _, _, encoded = _quantized(normalized)
    values, scales, _ = _quantized(normalized)
    actual_values = channel.scratch[13][: rows * 5120].view(rows, 5120)
    actual_scales = channel.scratch[14][: rows * 160].view(rows, 160)
    if not torch.equal(actual_values, values.view(torch.uint8)) or not torch.equal(
        actual_scales, scales
    ):
        raise AssertionError(
            "expert/shared input differs from unfloored MXFP8 row encoding"
        )
    mode = layer.persistent_tail.shared_a16
    lhs = normalized.double() if mode & 1 else encoded
    weight = layer.weights.experts.shared
    _dot(
        scratch.shared.gate_up,
        lhs,
        _weight(weight.gate_up.weight),
        extra=10,
        role="shared gate/up projection",
    )
    gate, up = scratch.shared.gate_up.double().chunk(2, dim=-1)
    gate = gate.clamp(max=10.0)
    up = up.clamp(-10.0, 10.0)
    exact = (gate / (1.0 + torch.exp(-gate))) * up
    # expf's FP32 relative error plus add/divide/multiply rounding; the final
    # BF16 rounding is bounded separately at its actual output spacing.
    _check(
        scratch.shared.intermediate,
        exact,
        _gamma(8) * exact.abs() + _rounding(scratch.shared.intermediate),
        "shared BF16 SwiGLU activation",
    )
    if mode & 2:
        lhs = scratch.shared.intermediate.double()
        stored = (
            channel.scratch[18][: rows * 576 * 2].view(torch.bfloat16).view(rows, 576)
        )
        if not torch.equal(stored, scratch.shared.intermediate):
            raise AssertionError(
                "shared down projection omits the BF16 activation boundary"
            )
    else:
        values, scales, lhs = _quantized(scratch.shared.intermediate)
        stored = channel.scratch[18][: rows * 576].view(rows, 576)
        stored_scales = channel.scratch[19][: rows * 20].view(rows, 20)[:, :18]
        if not torch.equal(stored, values.view(torch.uint8)) or not torch.equal(
            stored_scales, scales
        ):
            raise AssertionError(
                "shared down input differs from unfloored BF16-to-MXFP8 encoding"
            )
    _dot(
        scratch.shared.output,
        lhs,
        _weight(weight.down.weight)[:, :576],
        extra=1,
        role="shared down projection",
    )


def _routed(layer, channel, normalized):
    from .persistent_moe_reference import decode_weight

    scratch = layer.scratch
    rows = normalized.shape[0]
    routes = rows * 6
    representation = layer.weights.experts.experts._impl.representation_for("w4a8_mx")
    projected = (
        channel.scratch[27][: routes * 1152 * 2].view(torch.bfloat16).view(routes, 1152)
    )
    begin = 128 * 6 * 1152 * 2
    activated = (
        channel.scratch[27][begin : begin + routes * 576 * 2]
        .view(torch.bfloat16)
        .view(routes, 576)
    )
    gate, up = projected.double().chunk(2, dim=-1)
    gate = gate.clamp(max=10.0)
    up = up.clamp(-10.0, 10.0)
    exact = (gate / (1.0 + torch.exp(-gate))) * up
    _check(
        activated,
        exact,
        _gamma(8) * exact.abs() + _rounding(activated),
        "routed BF16 SwiGLU activation",
    )
    values, scales, intermediate = _quantized(activated, zero_scale=True)
    actual_values = channel.scratch[22][: routes * 576].view(routes, 576)
    actual_scales = channel.scratch[23][: routes * 20].view(routes, 20)[:, :18]
    if not torch.equal(actual_values, values.view(torch.uint8)) or not torch.equal(
        actual_scales, scales
    ):
        raise AssertionError(
            "routed activation differs from unfloored BF16-to-MXFP8 encoding"
        )
    _, _, decoded = _quantized(normalized)
    ids = scratch.router.ids.flatten()
    route_weights = scratch.router.weights.flatten().double()
    tokens = torch.arange(rows, device=ids.device).repeat_interleave(6)
    weighted = (
        channel.scratch[24][: routes * 5120 * 2].view(torch.bfloat16).view(routes, 5120)
    )
    for expert in sorted(set(ids.tolist())):
        selected = (ids == expert).nonzero().flatten()
        left = decoded[tokens[selected]]
        w13 = decode_weight(
            representation.w13_rp[expert],
            representation.w13_sfb[expert],
            rows=1152,
            columns=5120,
            group_rows=576,
        )
        # Canonical compact storage keeps up first and gate second; the
        # declared BF16 projection boundary is gate first, then up.
        w13 = torch.cat((w13[576:], w13[:576]), dim=0)
        _dot(projected[selected], left, w13, extra=10, role="routed gate/up projection")
        del w13
        w2 = decode_weight(
            representation.w2_rp[expert],
            representation.w2_sfb[expert],
            rows=5120,
            columns=576,
            group_rows=5120,
        )
        left = intermediate[selected]
        exact = (left @ w2.T) * route_weights[selected, None]
        magnitude = (left.abs() @ w2.abs().T) * route_weights[selected, None].abs()
        error = _gamma(577) * magnitude + _rounding(weighted[selected])
        _check(weighted[selected], exact, error, "weighted routed down projection")
        del w2
    reduced = torch.zeros((rows, 5120), dtype=torch.float32, device=weighted.device)
    weighted = weighted.view(rows, 6, 5120)
    for route in range(6):
        reduced.add_(weighted[:, route].float())
    if not torch.equal(scratch.routed, reduced.bfloat16()):
        raise AssertionError(
            "routed output omits ordered FP32 reduction of weighted BF16 routes"
        )


def _ffn_reduction(layer, channel):
    scratch = layer.scratch
    rows = scratch.mlp_output.shape[0]
    partial = (scratch.routed.float() + scratch.shared.output.float()).bfloat16()
    if not torch.equal(_local_inbox(channel, rows, 1), partial):
        raise AssertionError(
            "rank-local FFN sum differs from BF16 routed/shared addition"
        )
    epoch = int(channel.scratch[0].view(torch.int32).item())
    begin = 8192 + (4 + (epoch & 1)) * 128 * 5120 * 2
    total = torch.zeros((rows, 5120), dtype=torch.float32, device=partial.device)
    for rank in range(4):
        index = 0 if rank == channel.rank else rank + 1 if rank < channel.rank else rank
        values = (
            channel.peers[index][begin : begin + rows * 5120 * 2]
            .view(torch.bfloat16)
            .view(rows, 5120)
        )
        total.add_(values.float())
    if not torch.equal(scratch.mlp_output, total.bfloat16()):
        raise AssertionError(
            "FFN TP reduction differs from rank-ordered FP32 sum and BF16 output"
        )


def _gather(coordinator, value):
    peers = [None] * len(coordinator.ranks)
    dist.all_gather_object(peers, value, group=coordinator.group)
    return peers


def validate(layer, channel, *, expected_ids, coordinator, fixture):
    """Gate each actual operator before timing; whole-layer RMS is diagnostic.

    Every rank exchanges local errors before any peer-memory reference check.
    Validation reads explicit outputs and does not modify model state.
    """
    errors = []
    try:
        if layer.persistent_prologue is None:
            fixture.validate_numeric(layer)
        else:
            _prologue(layer, channel, fixture.active_case)
        if layer.persistent_tail is not None:
            _wo(layer, channel)
            normalized = _post_pre(layer, channel)
            _shared(layer, channel, normalized)
            _routed(layer, channel, normalized)
        if not torch.equal(layer.scratch.router.ids, expected_ids):
            raise AssertionError(
                "persistent candidate changes selected expert IDs for the restored input"
            )
    except Exception as error:
        errors.append(str(error))
    peers = _gather(coordinator, tuple(errors))
    if any(peers):
        raise AssertionError(
            f"persistent arithmetic validation failed on TP ranks: {peers}"
        )
    try:
        if layer.persistent_tail is not None:
            _attention_reduction(layer, channel)
            _ffn_reduction(layer, channel)
    except Exception as error:
        errors.append(str(error))
    peers = _gather(coordinator, tuple(errors))
    if any(peers):
        raise AssertionError(
            f"persistent transport validation failed on TP ranks: {peers}"
        )
