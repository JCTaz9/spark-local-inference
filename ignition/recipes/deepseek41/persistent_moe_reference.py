"""Preparation-only FP64 decoding of ordinary b12x compact MXFP4 experts.

The decoder restores logical rows and input columns from 128-row tiles,
including compact 64-row and 64-column tails without padding. It does not use the persistent decoder's weight-load logic.
Rows retain the native group order: FC1 ``w13`` has up rows before gate rows;
FC2 has ordinary output-row order. No checkpoint row rotation is undone.
"""

import torch


def _value_tiles(native, tile_rows, word_columns):
    """Restore logical rows/K words for equally sized native N tiles."""
    tiles = native.shape[0]
    parts = []
    full, tail = divmod(word_columns, 16)
    offset = 0
    for count, width in ((full, 16), (int(tail != 0), tail)):
        if not count:
            continue
        size = tile_rows * count * width
        block = native[:, offset : offset + size].reshape(
            tiles, count, width // 4, tile_rows // 32, 8, 4, 4
        )
        # Native inner word order is [K32, N32, row-within-8, K-word, N8].
        parts.append(
            block.permute(0, 3, 6, 4, 1, 2, 5).reshape(tiles * tile_rows, count * width)
        )
        offset += size
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def _scale_tiles(native, tile_rows, scale_columns):
    """Restore logical per-row K32 scales from native K128 scale tiles."""
    tiles = native.shape[0]
    parts = []
    full, tail = divmod(scale_columns, 4)
    offset = 0
    for count, width in ((full, 4), (int(tail != 0), tail)):
        if not count:
            continue
        size = tile_rows * count * width
        block = native[:, offset : offset + size].reshape(
            tiles, count, tile_rows, width
        )
        parts.append(
            block.permute(0, 2, 1, 3).reshape(tiles * tile_rows, count * width)
        )
        offset += size
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def _logical_rows(flat, *, rows, columns, group_rows, values):
    """Split compact row groups before undoing their independent N64 tails."""
    groups = []
    offset = 0
    restore = _value_tiles if values else _scale_tiles
    for _ in range(rows // group_rows):
        row_parts = []
        full, tail = divmod(group_rows, 128)
        for count, height in ((full, 128), (int(tail != 0), tail)):
            if not count:
                continue
            size = count * height * columns
            block = flat[offset : offset + size].reshape(count, height * columns)
            row_parts.append(restore(block, height, columns))
            offset += size
        groups.append(
            torch.cat(row_parts, dim=0) if len(row_parts) > 1 else row_parts[0]
        )
    return torch.cat(groups, dim=0) if len(groups) > 1 else groups[0]


def decode_weight(values_one_expert, scales_one_expert, *, rows, columns, group_rows):
    """Decode one native compact-N64 MXFP4 expert into FP64 ``[rows, columns]``.

    Both arguments are contiguous rank-one int32 storage views from the
    ordinary prepared b12x representation. Values contain eight E2M1 nibbles
    per word; scales contain four UE8M0 bytes per word, one scale per K32.
    The operation uses only ordinary Torch operations on the input device and
    never mutates the retained weight storage. UE8M0 byte 255 decodes as NaN.
    """
    dimensions = (rows, columns, group_rows)
    if any(type(value) is not int or value <= 0 or value % 64 for value in dimensions):
        raise ValueError("compact MXFP4 dimensions must be positive multiples of 64")
    if rows % group_rows:
        raise ValueError("compact MXFP4 row groups must cover every output row")
    for value, size, role in (
        (values_one_expert, rows * columns // 8, "values"),
        (scales_one_expert, rows * columns // 128, "scales"),
    ):
        if value.dtype != torch.int32 or value.ndim != 1 or not value.is_contiguous():
            raise ValueError(
                f"compact MXFP4 {role} must be contiguous rank-one int32 storage"
            )
        if value.numel() != size:
            raise ValueError(
                f"compact MXFP4 {role} require {size} int32 words, got {value.numel()}"
            )
    if values_one_expert.device != scales_one_expert.device:
        raise ValueError("compact MXFP4 values and scales must occupy the same device")
    words = _logical_rows(
        values_one_expert,
        rows=rows,
        columns=columns // 8,
        group_rows=group_rows,
        values=True,
    )
    scales = _logical_rows(
        scales_one_expert.view(torch.uint8),
        rows=rows,
        columns=columns // 32,
        group_rows=group_rows,
        values=False,
    )
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=words.device)
    codes = ((words[..., None] >> shifts) & 15).reshape(rows, columns)
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float64,
        device=words.device,
    )
    decoded_scales = torch.exp2(scales.to(torch.float64) - 127)
    decoded_scales = torch.where(scales == 255, float("nan"), decoded_scales)
    return (
        table[codes].reshape(rows, columns // 32, 32) * decoded_scales[..., None]
    ).reshape(rows, columns)
