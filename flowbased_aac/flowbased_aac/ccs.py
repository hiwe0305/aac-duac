"""Consensus-Coherent Chunk Selection (CCS): pick which of N sampled action chunks to execute.

Training-free. The policy is sampled N times from the same observation (different initial flow noise); instead of
executing sample 0 (what AAC does) CCS executes the candidate that

  1. agrees with the majority on the gripper open/close sequence over the executed prefix (gripper vote),
  2. lies in the middle of the remaining candidates (consensus: smallest mean distance to the others, a medoid), and
  3. continues the not-yet-executed tail of the previous chunk (coherence: small jump at the chunk boundary).

All distances are on the executed prefix of length h (the actions that will actually run), on absolute end-effector
targets: position in metres plus ``rot_weight`` metres per radian of rotation (geodesic angle between orientations).

    keep  C = {j : Hamming(g_j[:h], g*) <= gripper_tol}        g* = per-step majority gripper state
    D_j   = mean_{k in C, k != j} d(a_j[:h], a_k[:h])
    B_j   = d(a_j[:L], prev[h_prev : h_prev + L]),  L = min(h, H - h_prev)   (0 when there is no previous chunk)
    j*    = argmin_{j in C}  z(D_j) + lam * z(B_j)           z = standardisation within C

``lam = 0`` gives a pure medoid; ``gripper_tol = None`` disables the vote.

Action spaces (``action_space``):
  "absolute"      VLABench: [target pos (m), target euler xyz (rad), gripper], open iff gripper >= 0.1
  "libero_delta"  LIBERO OSC: per-step [dpos, daxis-angle, gripper] in robosuite units (1.0 = 0.05 m / 0.5 rad),
                  gripper -1 open / +1 close. Chunks are integrated into a trajectory relative to the current pose
                  (cumulative sums, rotation as an accumulated rotation vector) before measuring distances; the
                  previous chunk's tail is integrated from the same origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

GRIPPER_OPEN_THRESHOLD = 0.1  # VLABench official client: fingers open iff gripper >= 0.1
LIBERO_POS_UNIT, LIBERO_ROT_UNIT = 0.05, 0.5  # robosuite OSC_POSE: input 1.0 -> 0.05 m / 0.5 rad


@dataclass
class CCSConfig:
    lam: float = 1.0            # weight of coherence relative to consensus
    rot_weight: float = 0.05    # metres per radian
    gripper_tol: Optional[int] = 1  # None: no gripper vote
    use_consensus: bool = True


@dataclass
class CCSResult:
    index: int
    kept: List[int]
    consensus: List[float]   # D_j for all N (nan if dropped by the vote)
    coherence: List[float]   # B_j for all N (nan if dropped)
    score: List[float]
    overlap: int
    gripper_majority: List[int] = field(default_factory=list)

    def as_log(self) -> Dict[str, object]:
        r = lambda xs: [None if not np.isfinite(x) else round(float(x), 5) for x in xs]
        return {"ccs_index": self.index, "ccs_kept": len(self.kept), "ccs_D": r(self.consensus),
                "ccs_B": r(self.coherence), "ccs_score": r(self.score), "ccs_overlap": self.overlap}


def _pose_distance(a: np.ndarray, b: np.ndarray, rot_weight: float, rot_mode: str = "euler") -> np.ndarray:
    """Mean over steps of position L2 + rot_weight * rotation distance. a, b: (..., T, 7) broadcastable.

    rot_mode "euler": geodesic angle between euler-xyz orientations; "vector": L2 between rotation vectors."""

    a, b = np.broadcast_arrays(a, b)
    pos = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)
    if rot_weight > 0 and rot_mode == "vector":
        pos = pos + rot_weight * np.linalg.norm(a[..., 3:6] - b[..., 3:6], axis=-1)
    elif rot_weight > 0:
        shape = a.shape[:-1]
        ra = Rotation.from_euler("xyz", a[..., 3:6].reshape(-1, 3))
        rb = Rotation.from_euler("xyz", b[..., 3:6].reshape(-1, 3))
        ang = (ra.inv() * rb).magnitude().reshape(shape)
        pos = pos + rot_weight * ang
    return pos.mean(axis=-1)


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-12 else (x - x.mean()) / sd


def _integrate_libero(deltas: np.ndarray) -> np.ndarray:
    """(..., T, 7) LIBERO deltas -> (..., T, 7) trajectory relative to the current pose (m, rad rotvec, gripper)."""

    out = np.array(deltas, dtype=np.float64, copy=True)
    out[..., :3] = np.cumsum(deltas[..., :3], axis=-2) * LIBERO_POS_UNIT
    out[..., 3:6] = np.cumsum(deltas[..., 3:6], axis=-2) * LIBERO_ROT_UNIT
    return out


def select_chunk(samples: np.ndarray, h: int, prev_chunk: Optional[np.ndarray] = None, prev_h: int = 0,
                 cfg: CCSConfig = CCSConfig(), action_space: str = "absolute") -> CCSResult:
    """samples: (N, H, 7) actions; prev_chunk: (H, 7) chunk whose first ``prev_h`` actions were executed."""

    samples = np.asarray(samples, dtype=np.float64)[..., :7]
    n, horizon, _ = samples.shape
    h = int(max(1, min(h, horizon)))
    if action_space == "libero_delta":
        is_open = samples[:, :h, 6] < 0.0
        rot_mode = "vector"
        if prev_chunk is not None:
            prev_chunk = np.asarray(prev_chunk, dtype=np.float64)[:, :7]
            tail = _integrate_libero(prev_chunk[prev_h:])
            prev_chunk = np.concatenate([np.zeros((prev_h, 7)), tail], axis=0)  # tail indexed from prev_h, origin = now
        samples = _integrate_libero(samples)
    elif action_space == "absolute":
        is_open = samples[:, :h, 6] >= GRIPPER_OPEN_THRESHOLD
        rot_mode = "euler"
    else:
        raise ValueError(f"unknown action_space {action_space!r}")
    pre = samples[:, :h]

    grip = is_open.astype(int)  # (N, h)
    majority = (grip.mean(axis=0) >= 0.5).astype(int)
    if cfg.gripper_tol is None:
        kept = np.arange(n)
    else:
        ham = np.abs(grip - majority[None]).sum(axis=1)
        kept = np.flatnonzero(ham <= cfg.gripper_tol)
        if kept.size == 0:  # per-step majority matches nobody: keep the closest ones
            kept = np.flatnonzero(ham == ham.min())

    D = np.full(n, np.nan)
    if cfg.use_consensus and kept.size > 1:
        sub = pre[kept]
        pair = _pose_distance(sub[:, None], sub[None, :], cfg.rot_weight, rot_mode)  # (k, k)
        D[kept] = pair.sum(axis=1) / (kept.size - 1)
    else:
        D[kept] = 0.0

    B = np.full(n, np.nan)
    overlap = 0
    if prev_chunk is not None and cfg.lam > 0:
        prev_chunk = np.asarray(prev_chunk, dtype=np.float64)
        overlap = int(max(0, min(h, prev_chunk.shape[0] - prev_h)))
    if overlap > 0:
        tail = prev_chunk[prev_h:prev_h + overlap]
        B[kept] = _pose_distance(pre[kept, :overlap], tail[None], cfg.rot_weight, rot_mode)
    else:
        B[kept] = 0.0

    score = np.full(n, np.nan)
    score[kept] = _z(D[kept]) + cfg.lam * _z(B[kept])
    best = int(kept[np.argmin(score[kept])])  # ties -> lowest index (sample 0 first)
    return CCSResult(index=best, kept=kept.tolist(), consensus=D.tolist(), coherence=B.tolist(),
                     score=score.tolist(), overlap=overlap, gripper_majority=majority.tolist())


VARIANTS = {
    "ccs": CCSConfig(),
    "medoid": CCSConfig(lam=0.0, gripper_tol=None),
    "coherent": CCSConfig(lam=1.0, gripper_tol=None, use_consensus=False),
    "gripvote": CCSConfig(lam=0.0, use_consensus=False),
}


def _selftest() -> None:
    rng = np.random.default_rng(0)
    base = np.zeros((10, 7)); base[:, 0] = np.linspace(0, 0.09, 10); base[:, 3] = np.pi; base[:, 6] = 1.0
    s = base[None] + rng.normal(0, 0.002, (20, 10, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0]
    s[3, :, 1] += 0.08                       # outlier path
    s[5, 2:, 6] = 0.0                        # closes the gripper early (minority)
    r = select_chunk(s, 8)
    assert r.index not in (3, 5) and 5 not in r.kept, r.index
    assert np.nanargmax(r.consensus) == 3
    prev = base.copy(); prev[:, 1] = 0.05    # previous plan went the other way: coherence prefers outlier-side samples
    s2 = s.copy(); s2[3, :, 6] = 1.0
    r2 = select_chunk(s2, 8, prev_chunk=prev, prev_h=8, cfg=CCSConfig(lam=10.0))
    assert r2.index == 3 and r2.overlap == 2, (r2.index, r2.overlap)
    r3 = select_chunk(s, 8, cfg=VARIANTS["medoid"])
    assert r3.index != 3
    assert select_chunk(np.repeat(base[None], 4, 0), 8).index == 0  # identical samples -> sample 0
    euler = np.array([[np.pi - 1e-3, 0, 0], [-np.pi + 1e-3, 0, 0]])  # wrap-around: nearly the same orientation
    a = np.zeros((1, 7)); b = np.zeros((1, 7)); a[0, 3:6] = euler[0]; b[0, 3:6] = euler[1]
    assert _pose_distance(a, b, 1.0) < 0.01
    # LIBERO deltas: same scenario expressed as per-step deltas in robosuite units, gripper -1 open / +1 close
    d = np.diff(np.concatenate([np.zeros((20, 1, 7)), s], axis=1), axis=1)
    d[..., :3] /= LIBERO_POS_UNIT; d[..., 3:6] = 0.0
    d[..., 6] = np.where(s[..., 6] >= GRIPPER_OPEN_THRESHOLD, -1.0, 1.0)
    rl = select_chunk(d, 8, action_space="libero_delta")
    assert rl.index not in (3, 5) and 5 not in rl.kept and np.nanargmax(rl.consensus) == 3, rl.index
    assert np.allclose(_integrate_libero(d)[..., :3], s[..., :3])
    prev_d = np.zeros((10, 7)); prev_d[:, 1] = 0.05 / LIBERO_POS_UNIT; prev_d[:, 6] = -1.0  # previous plan heads +y
    d2 = d.copy(); d2[3, :, 6] = -1.0
    rl2 = select_chunk(d2, 8, prev_chunk=prev_d, prev_h=8, cfg=CCSConfig(lam=10.0), action_space="libero_delta")
    assert rl2.index == 3 and rl2.overlap == 2, (rl2.index, rl2.overlap)
    print("[PASS] ccs: gripper vote drops minority, medoid avoids outlier, coherence follows previous plan, "
          "identical samples -> 0, euler wrap safe; same behaviour on LIBERO delta actions")


if __name__ == "__main__":
    _selftest()
