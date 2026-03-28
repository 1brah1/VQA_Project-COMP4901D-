from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compress_27x27_tokens(x: torch.Tensor, *, target_tokens: int) -> torch.Tensor:
    """
    Deterministic compression for square patch grids via adaptive average pooling.

    Backward-compatible function name retained.
    Supports both 27x27 (729) and 24x24 (576) patch grids commonly seen with SigLIP.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (B, N, D); got {tuple(x.shape)}")
    b, n, d = x.shape
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"Expected square number of patch tokens; got {n}")

    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0; got {target_tokens}")
    if target_tokens > n:
        raise ValueError(f"target_tokens cannot exceed input tokens ({n}); got {target_tokens}")
    if target_tokens == n:
        return x

    out_hw = _best_output_hw(target_tokens=target_tokens, in_side=side)
    if out_hw is None:
        supported = ", ".join(str(t) for t in recommended_targets(n))
        raise ValueError(
            f"Cannot map {n} input tokens to target_tokens={target_tokens} "
            f"with a spatially valid grid. Recommended targets for {n}: [{supported}]"
        )

    out_h, out_w = out_hw
    grid = x.view(b, side, side, d).permute(0, 3, 1, 2)  # (B,D,H,W)
    pooled = F.adaptive_avg_pool2d(grid, output_size=(out_h, out_w))  # (B,D,out_h,out_w)
    return pooled.permute(0, 2, 3, 1).reshape(b, out_h * out_w, d)  # (B,target,D)


def recommended_targets(num_tokens: int) -> list[int]:
    """
    Stable benchmark presets for known SigLIP grid sizes.
    """
    if num_tokens == 729:
        return [729, 243, 81, 27, 9]
    if num_tokens == 576:
        return [576, 192, 81, 36, 9]
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        return [num_tokens]
    candidates = [num_tokens, 81, 64, 49, 36, 25, 16, 9, 4]
    out = []
    for t in candidates:
        if t <= num_tokens and _best_output_hw(target_tokens=t, in_side=side) is not None:
            out.append(t)
    return out or [num_tokens]


def _best_output_hw(*, target_tokens: int, in_side: int) -> Optional[Tuple[int, int]]:
    pairs = []
    for h in range(1, in_side + 1):
        if target_tokens % h != 0:
            continue
        w = target_tokens // h
        if w <= in_side:
            pairs.append((h, w))
    if not pairs:
        return None
    # Prefer near-square output for balanced spatial detail.
    return min(pairs, key=lambda hw: abs(hw[0] - hw[1]))

