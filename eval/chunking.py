"""Horizon selection helpers for pi0.5 (Fixed-k, AAC Gaussian/Bernoulli entropy, and DUAC JVP divergence)."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FLOWBASED_ROOT = ROOT / "flowbased_aac"
if str(FLOWBASED_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOWBASED_ROOT))

from flowbased_aac.duac_policy import (  # noqa: E402
    DUACConfig,
    DUACDecisionResult,
    action_magnitude_pi05,
    compose_rotations_euler,
    find_first_idx,
    prefix_mean_jump_horizon,
    select_duac_horizon,
)


def fixed_horizon(k: int = 10, max_horizon: int = 10) -> int:
    """Return clamped fixed horizon k in {1..max_horizon}."""
    return max(1, min(int(k), int(max_horizon)))


__all__ = [
    "DUACConfig",
    "DUACDecisionResult",
    "action_magnitude_pi05",
    "compose_rotations_euler",
    "find_first_idx",
    "fixed_horizon",
    "prefix_mean_jump_horizon",
    "select_duac_horizon",
]
