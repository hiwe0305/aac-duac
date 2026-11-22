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
                  gripper -1 open / +1 close. Chunks are integrated into a trajectory relative to the current pose:
                  positions by cumulative sums, orientations by composing the rotations the way robosuite's OSC sets
                  its goal (control_utils.set_goal_orientation: R_goal = Exp(delta) @ R_current, i.e. world-frame
                  left multiplication), stored as rotation vectors; orientation distances are geodesic on SO(3).
                  The previous chunk's tail is integrated from the same origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
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


ACTION_SPACES = ("absolute", "libero_delta")
_ROT_MODE = {"absolute": "euler", "libero_delta": "rotvec"}


def _gripper_open(actions: np.ndarray, action_space: str) -> np.ndarray:
    """Boolean open state per step: VLABench open iff >= 0.1; LIBERO -1 open / +1 close."""

    g = np.asarray(actions)[..., 6]
    return g < 0.0 if action_space == "libero_delta" else g >= GRIPPER_OPEN_THRESHOLD


def _rotations(x: np.ndarray, rot_mode: str) -> Rotation:
    x = np.ascontiguousarray(x).reshape(-1, 3)
    return Rotation.from_euler("xyz", x) if rot_mode == "euler" else Rotation.from_rotvec(x)


def _step_distance(a: np.ndarray, b: np.ndarray, rot_weight: float, rot_mode: str = "euler") -> np.ndarray:
    """Per-step distance: position L2 (m) + rot_weight (m/rad) * geodesic angle between the two orientations.
    a, b: (..., T, 7) broadcastable; rot_mode "euler" (euler xyz, wrap-safe) or "rotvec" (rotation vectors)."""

    a, b = np.broadcast_arrays(a, b)
    d = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)
    if rot_weight > 0:
        if rot_mode not in ("euler", "rotvec"):
            raise ValueError(f"unknown rot_mode {rot_mode!r}")
        ang = (_rotations(a[..., 3:6], rot_mode).inv() * _rotations(b[..., 3:6], rot_mode)).magnitude()
        d = d + rot_weight * ang.reshape(a.shape[:-1])
    return d


def _pose_distance(a: np.ndarray, b: np.ndarray, rot_weight: float, rot_mode: str = "euler") -> np.ndarray:
    """Mean over steps of :func:`_step_distance`."""

    return _step_distance(a, b, rot_weight, rot_mode).mean(axis=-1)


Z_MIN_SD = 1e-6  # 1 micron: below this spread the candidates are treated as equivalent (no z-score amplification)


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return np.zeros_like(x) if sd < Z_MIN_SD else (x - x.mean()) / sd


def _validate(samples: np.ndarray, prev_h: int = 0, action_space: str = "absolute",
              prev: Optional[np.ndarray] = None) -> np.ndarray:
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 3 or s.shape[0] < 1 or s.shape[1] < 1 or s.shape[2] < 7:
        raise ValueError(f"samples must be (N >= 1, H >= 1, >= 7), got {s.shape}")
    if not np.all(np.isfinite(s[..., :7])):
        raise ValueError("samples contain NaN or inf")
    if action_space not in ACTION_SPACES:
        raise ValueError(f"unknown action_space {action_space!r} (expected one of {ACTION_SPACES})")
    if prev is not None:
        p = np.asarray(prev, dtype=np.float64)
        if p.shape[-1] < 7 or not np.all(np.isfinite(p[..., :7])):
            raise ValueError(f"previous chunk/samples invalid: shape {p.shape} or non-finite values")
        if not 0 <= int(prev_h) <= p.shape[-2]:
            raise ValueError(f"prev_h={prev_h} outside [0, {p.shape[-2]}]")
    return s[..., :7]


def _integrate_libero(deltas: np.ndarray) -> np.ndarray:
    """(..., T, 7) LIBERO deltas -> (..., T, 7) trajectory relative to the current pose.

    Position: cumulative sum x 0.05 m. Orientation: R_t = Exp(0.5 d_t) @ ... @ Exp(0.5 d_1) (robosuite OSC goal update),
    stored as a rotation vector (angle <= pi). Gripper unchanged."""

    d = np.asarray(deltas, dtype=np.float64)
    out = np.array(d, copy=True)
    out[..., :3] = np.cumsum(d[..., :3], axis=-2) * LIBERO_POS_UNIT
    lead, T = d.shape[:-2], d.shape[-2]
    rv = d[..., 3:6].reshape(-1, T, 3) * LIBERO_ROT_UNIT
    acc = Rotation.identity(rv.shape[0])
    comp = np.empty_like(rv)
    for t in range(T):
        acc = Rotation.from_rotvec(rv[:, t]) * acc  # world-frame left multiplication
        comp[:, t] = acc.as_rotvec()
    out[..., 3:6] = comp.reshape(*lead, T, 3)
    return out


def select_chunk(samples: np.ndarray, h: int, prev_chunk: Optional[np.ndarray] = None, prev_h: int = 0,
                 cfg: CCSConfig = CCSConfig(), action_space: str = "absolute") -> CCSResult:
    """samples: (N, H, 7) actions; prev_chunk: (H, 7) chunk whose first ``prev_h`` actions were executed."""

    samples = _validate(samples, prev_h, action_space, prev_chunk)
    n, horizon, _ = samples.shape
    h = int(max(1, min(h, horizon)))
    rot_mode = _ROT_MODE[action_space]
    is_open = _gripper_open(samples[:, :h], action_space)
    if action_space == "libero_delta":
        if prev_chunk is not None:
            prev_chunk = np.asarray(prev_chunk, dtype=np.float64)[:, :7]
            tail = _integrate_libero(prev_chunk[prev_h:])
            prev_chunk = np.concatenate([np.zeros((prev_h, 7)), tail], axis=0)  # tail indexed from prev_h, origin = now
        samples = _integrate_libero(samples)
    pre = samples[:, :h]

    grip = is_open.astype(int)  # (N, h)
    frac = grip.mean(axis=0)
    majority = (frac > 0.5).astype(int)
    decided = frac != 0.5  # exact ties (e.g. 10/10) express no majority: those steps are not voted on
    if cfg.gripper_tol is None:
        kept = np.arange(n)
    else:
        ham = (np.abs(grip - majority[None]) * decided[None]).sum(axis=1)
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


@dataclass
class EGCResult:
    index: int
    commit: bool     # True: stayed with the previous plan's mode; False: switched to the global medoid
    support: int     # fresh samples whose overlap window agrees with the previous plan's unexecuted tail
    peak: int        # largest agreement count around any sample (mass of the best-supported mode)
    n_cont: int      # size of the continuation set used for the in-mode medoid
    overlap: int
    radius: float

    def as_log(self) -> Dict[str, object]:
        return {"egc_index": self.index, "egc_commit": self.commit, "egc_support": self.support, "egc_peak": self.peak,
                "egc_n_cont": self.n_cont, "egc_overlap": self.overlap, "egc_radius": round(float(self.radius), 5)}


def select_egc(samples: np.ndarray, h: int, prev_chunk: Optional[np.ndarray] = None, prev_h: int = 0,
               gamma: float = 0.5, rot_weight: float = 0.05, action_space: str = "absolute",
               knn_frac: float = 0.2, link_mult: float = 2.0) -> EGCResult:
    """Evidence-Gated Commitment: stay in the previous plan's mode while the fresh samples still support it.

    The previous plan is represented by its unexecuted tail, which covers the first L = H - h_prev steps of the new
    chunk (same time steps). On that overlap window:
      s_L      = median over samples of the distance to their k-th nearest other sample, k = max(1, round(knn_frac
                 (N - 1))) (= 4 for N = 20): a LOCAL (within-mode) scale.
      modes    = connected components of the graph linking samples with d_L <= link_mult * s_L (single linkage):
                 well-separated modes become separate components, a broad unimodal cloud stays one component.
      peak     = size of the largest mode                          mass of the best-supported mode
      support  = total size of the modes containing a sample within link_mult * s_L of the tail
                                                                   mass that still agrees with the previous plan
      commit  <=> support >= gamma * peak
      commit -> medoid (distance on the executed prefix h) of the samples in the tail's mode(s)
      switch -> medoid of all N samples
    Mode masses are counted after clustering, not with a global radius: the median of all pairwise distances (first
    version) approaches the distance between the modes when they are separated, since most pairs are cross-mode
    (96 / 190 for a 12/8 split), and a bare local radius turns the counts into noisy local densities.
    With no previous chunk or no overlap the global medoid is executed. Unlike a soft coherence cost (BID backward
    coherence, CCS) the previous plan is never followed against the evidence: once fewer than gamma x peak samples
    agree with it, the policy switches; unlike the memoryless medoid it does not flip between two comparably supported
    modes at every replan.
    """

    samples = _validate(samples, prev_h, action_space, prev_chunk)
    n, horizon, _ = samples.shape
    h = int(max(1, min(h, horizon)))
    rot_mode = _ROT_MODE[action_space]
    if action_space == "libero_delta":
        traj = _integrate_libero(samples)
        if prev_chunk is not None:
            prev_chunk = np.asarray(prev_chunk, dtype=np.float64)[:, :7]
            prev_chunk = np.concatenate([np.zeros((prev_h, 7)), _integrate_libero(prev_chunk[prev_h:])], axis=0)
    else:
        traj = samples
        if prev_chunk is not None:
            prev_chunk = np.asarray(prev_chunk, dtype=np.float64)[:, :7]
    pre = traj[:, :h]
    dfull = _pose_distance(pre[:, None], pre[None, :], rot_weight, rot_mode)
    medoid = int(np.argmin(dfull.sum(axis=1)))
    L = int(max(0, min(horizon, prev_chunk.shape[0] - prev_h))) if prev_chunk is not None else 0
    if L == 0 or n < 2:
        return EGCResult(medoid, False, 0, 0, 0, 0, 0.0)
    win = traj[:, :L]
    dL = _pose_distance(win[:, None], win[None, :], rot_weight, rot_mode)
    k = int(max(1, min(n - 1, round(knn_frac * (n - 1)))))
    knn = np.sort(dL + np.diag(np.full(n, np.inf)), axis=1)[:, k - 1]  # k-th nearest OTHER sample
    r = float(link_mult * np.median(knn))
    _, label = connected_components(csr_matrix(dL <= r), directed=False)
    sizes = np.bincount(label)
    peak = int(sizes.max())
    dtail = _pose_distance(win, prev_chunk[prev_h:prev_h + L][None], rot_weight, rot_mode)
    tail_modes = np.unique(label[dtail <= r])
    cont = np.flatnonzero(np.isin(label, tail_modes))
    support = int(cont.size)
    if support == 0 or support < gamma * peak:
        return EGCResult(medoid, False, support, peak, 0, L, r)
    best = int(cont[np.argmin(dfull[np.ix_(cont, cont)].sum(axis=1))])
    return EGCResult(best, True, support, peak, support, L, r)


def select_xdc(samples: np.ndarray, h: int, prev_samples: Optional[np.ndarray] = None, prev_h: int = 0,
               lam: float = 1.0, rot_weight: float = 0.05, action_space: str = "absolute") -> Dict[str, object]:
    """Cross-Decision Consensus: the sample most central w.r.t. the current samples AND the previous decision's samples.

      C_j = mean_k d_h(a_j, a_k)                                  (current consensus, executed prefix h)
      P_j = mean_m d_L(a_j[:L], b_m[h_prev : h_prev + L])          (agreement with the previous distribution on the
                                                                    overlapping time steps, L = H - h_prev)
      j*  = argmin_j  C_j / median(C) + lam * P_j / median(P)
    A heuristic: a weighted sum of median-normalised mean distances (NOT a product of densities). Intended effect:
    transient, observation-driven flips of the policy's distribution are damped by the previous distribution, while a
    persistent change is followed one decision later (the previous distribution has moved by then). Unlike coherence
    with the single executed chunk (BID, CCS) it does not lock onto one possibly atypical chunk. Without previous
    samples it is the medoid.
    """

    samples = _validate(samples, prev_h, action_space, prev_samples)
    n, horizon, _ = samples.shape
    h = int(max(1, min(h, horizon)))
    delta = action_space == "libero_delta"
    rot_mode = _ROT_MODE[action_space]
    traj = _integrate_libero(samples) if delta else samples
    C = _pose_distance(traj[:, None, :h], traj[None, :, :h], rot_weight, rot_mode).mean(axis=1)
    L = int(max(0, horizon - prev_h)) if prev_samples is not None else 0
    if L == 0:
        j = int(np.argmin(C))
        return {"index": j, "xdc_overlap": 0}
    prev = np.asarray(prev_samples, dtype=np.float64)[..., :7]
    tails = _integrate_libero(prev[:, prev_h:prev_h + L]) if delta else prev[:, prev_h:prev_h + L]
    P = _pose_distance(traj[:, None, :L], tails[None], rot_weight, rot_mode).mean(axis=1)
    score = C / max(np.median(C), 1e-12) + lam * P / max(np.median(P), 1e-12)
    j = int(np.argmin(score))
    return {"index": j, "xdc_overlap": L, "xdc_medoid": int(np.argmin(C))}


def select_keystone(samples: np.ndarray, tau: float = 0.3, n_clusters: int = 2, max_iters: int = 10,
                    executed_steps: Optional[int] = None) -> Dict[str, object]:
    """KeyStone baseline (Dai et al., arXiv 2605.08638; repo dywsjtu/keystone, cluster_medoid_select defaults).

    Pass the policy's native output space (e.g. the server's ``normalized_action``), as KeyStone does: flattened L2
    on absolute euler angles is wrong whenever samples straddle +-pi (63% of VLABench decisions).

    x_i = flattened chunk (first ``executed_steps`` steps, all if None); Delta_ij = ||x_i - x_j||_2.
    Global medoid m; spread s = ||mean(x) - x_m|| / (median_{i<j} Delta_ij + 1e-8). If s < tau return m (unimodal).
    Otherwise k-means with C clusters (deterministic init = first C points, as in the repo; max 10 iterations, empty
    cluster keeps its centroid) and return the medoid of the largest cluster.
    """

    x = _validate(samples)
    if executed_steps:
        x = x[:, :executed_steps]
    x = x.reshape(len(x), -1)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    m = int(np.argmin(d.sum(axis=1)))
    med = float(np.median(d[np.triu_indices(len(x), 1)]))
    s = float(np.linalg.norm(x.mean(axis=0) - x[m]) / (med + 1e-8))
    if s < tau:
        return {"index": m, "ks_spread": round(s, 4), "ks_clustered": False}
    cent = x[:n_clusters].copy()
    for _ in range(max_iters):
        z = np.argmin(np.linalg.norm(x[:, None] - cent[None], axis=-1), axis=1)
        new = np.array([x[z == c].mean(axis=0) if np.any(z == c) else cent[c] for c in range(n_clusters)])
        if np.allclose(new, cent, atol=1e-6):
            break
        cent = new
    z = np.argmin(np.linalg.norm(x[:, None] - cent[None], axis=-1), axis=1)
    big = int(np.bincount(z, minlength=n_clusters).argmax())
    members = np.flatnonzero(z == big)
    i = int(members[np.argmin(d[np.ix_(members, members)].sum(axis=1))])
    return {"index": i, "ks_spread": round(s, 4), "ks_clustered": True, "ks_cluster_size": int(members.size)}


def rate_limit_position(target: np.ndarray, last_cmd: np.ndarray, max_step: float) -> np.ndarray:
    """Clip the commanded position change per control step to ``max_step`` (m); orientation and gripper untouched."""

    out = np.array(target, dtype=np.float64, copy=True)
    dp = out[:3] - last_cmd[:3]
    n = float(np.linalg.norm(dp))
    if n > max_step:
        out[:3] = last_cmd[:3] + dp * (max_step / n)
    return out


def consensus_horizon(samples: np.ndarray, index: int, radius: float = 0.02, quorum: float = 0.5,
                      min_h: int = 2, rot_weight: float = 0.05, action_space: str = "absolute") -> int:
    """How many steps of the chosen chunk the other samples still agree on (uncertainty of the executed chunk).

    Sample j agrees with the chosen chunk c at step k iff (i) the full pose agrees: position distance + rot_weight x
    geodesic orientation angle <= ``radius`` (m; 2 cm <-> 0.4 rad at 0.05 m/rad) and (ii) the gripper state (open /
    closed) is the same. agree_k = fraction of the N - 1 OTHER samples (the chosen chunk is not counted) that agree at
    step k; k* = the last step before agree_k first drops below ``quorum``; returns max(min_h, k*). Unlike AAC's marginal
    entropy over all samples or DUAC's divergence of one random sample, it measures agreement with the chunk that will
    actually be executed.
    """

    samples = _validate(samples, 0, action_space)
    n, horizon, _ = samples.shape
    if not 0 <= index < n:
        raise ValueError(f"index {index} outside [0, {n})")
    if n == 1:
        return horizon
    traj = _integrate_libero(samples) if action_space == "libero_delta" else samples
    d = _step_distance(traj, traj[index][None], rot_weight, _ROT_MODE[action_space])  # (N, H)
    g = _gripper_open(samples, action_space)
    agree = (d <= radius) & (g == g[index][None])
    others = np.arange(n) != index
    low = agree[others].mean(axis=0) < quorum
    k = int(np.argmax(low)) if low.any() else horizon
    return int(max(min_h, min(k, horizon)))


VARIANTS = {
    "ccs": CCSConfig(),
    "medoid": CCSConfig(lam=0.0, gripper_tol=None),
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
    # two modes: 12 samples go along x (L), 8 are offset in +y from the first step on (R)
    rng2 = np.random.default_rng(1)
    L = base.copy(); R = base.copy(); R[:, 1] = np.linspace(0.02, 0.06, 10)
    fresh = np.concatenate([L[None] + rng2.normal(0, 0.002, (12, 10, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0],
                            R[None] + rng2.normal(0, 0.002, (8, 10, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0]])
    # EGC on two well-separated modes: 12 along x (mA), 8 offset by 6 cm in y (mB)
    r12 = np.random.default_rng(3)
    mA = base[None] + r12.normal(0, 3e-4, (12, 10, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0]
    mB = base[None] + r12.normal(0, 3e-4, (8, 10, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0]; mB[..., 1] += 0.06
    two = np.concatenate([mA, mB]); prevB = np.concatenate([np.zeros((8, 7)), mB[0, :2]])
    e1 = select_egc(two, 8, prevB, 8)                      # B keeps 8 vs A's 12 -> 8 >= 0.5 * peak -> stay in B
    assert e1.commit and e1.index >= 12, e1
    e2 = select_egc(two, 8, prevB, 8, gamma=0.9)           # stricter gate -> switch to the global medoid (A)
    assert not e2.commit and e2.index < 12, e2
    e3 = select_egc(np.concatenate([mA, mA[:8]]), 8, prevB, 8)  # B lost all support -> switch
    assert not e3.commit and e3.support == 0, e3
    assert select_egc(two, 8).commit is False              # first decision -> medoid
    dB = np.diff(np.concatenate([np.zeros((20, 1, 7)), two], axis=1), axis=1); dB[..., :3] /= LIBERO_POS_UNIT
    dB[..., 3:6] = 0; dB[..., 6] = -1.0
    pB = np.zeros((10, 7)); pB[8:] = np.diff(np.concatenate([np.zeros((1, 7)), mB[0, :2]], axis=0), axis=0)
    pB[:, :3] /= LIBERO_POS_UNIT; pB[:, 3:6] = 0; pB[:, 6] = -1.0
    e4 = select_egc(dB, 8, pB, 8, action_space="libero_delta")
    assert e4.commit and e4.index >= 12, e4
    x0 = select_xdc(fresh, 8)
    assert x0["index"] == int(np.argmin(_pose_distance(fresh[:, None, :8], fresh[None, :, :8], 0.05).mean(1)))
    prevS = R[None] + rng2.normal(0, 0.002, (20, 10, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0]  # previous distribution was all R
    prevS[:, 8:] = R[None, :2] + rng2.normal(0, 0.002, (20, 2, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0]
    assert select_xdc(fresh, 8, prevS, 8, lam=5.0)["index"] >= 12  # strong temporal term keeps mode R
    split = np.repeat(base[None], 20, 0); split[12:, 5:, 1] += 0.05  # 12 samples agree all along, 8 leave at step 6
    assert consensus_horizon(split, 0) == 10                        # 12/20 >= 0.5 agree on every step
    split[4:12, 5:, 1] += 0.05                                      # now only 4/20 agree after step 5
    assert consensus_horizon(split, 0) == 5 and consensus_horizon(split, 0, quorum=0.1) == 10
    ks = select_keystone(fresh)                                     # bimodal 12/8 -> clustered, medoid of mode L
    assert ks["ks_clustered"] and ks["index"] < 12, ks
    one = np.repeat(L[None], 3, 0); one[1, :, 1] += 0.01; one[2, :, 1] -= 0.01  # symmetric: medoid = mean -> guard
    assert select_keystone(one)["ks_clustered"] is False and select_keystone(one)["index"] == 0
    lc = np.zeros(7); tgt = np.zeros(7); tgt[0] = 0.05; tgt[6] = 1.0
    rl = rate_limit_position(tgt, lc, 0.014)
    assert abs(np.linalg.norm(rl[:3]) - 0.014) < 1e-9 and rl[6] == 1.0
    assert np.allclose(rate_limit_position(tgt * 0.1, lc, 0.014), tgt * 0.1)
    # --- review fixes (2026-11-22) ---
    # EGC link radius is a within-mode scale (old median-of-all-pairs radius was ~6 cm here); masses = 8 vs 12
    assert e1.radius < 0.005 and e1.support == 8 and e1.peak == 12, e1
    broad = base[None] + r12.normal(0, 0.003, (20, 10, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0]  # one broad mode
    eb = select_egc(broad, 8, np.concatenate([np.zeros((8, 7)), broad[0, :2]]), 8)
    assert eb.peak == 20 and eb.commit, eb                  # a broad unimodal cloud is a single mode
    # SO(3) geodesic on rotation vectors: +-(pi - eps) about the same axis are the same orientation
    va = np.zeros((1, 7)); vb = np.zeros((1, 7)); va[0, 3] = np.pi - 1e-4; vb[0, 3] = -(np.pi - 1e-4)
    assert _pose_distance(va, vb, 0.05, "rotvec") < 1e-4
    # LIBERO rotation composition follows robosuite: R_2 = Exp(0.5 d_2) @ Exp(0.5 d_1)
    dd = np.zeros((1, 2, 7)); dd[0, 0, 3] = 1.0; dd[0, 1, 4] = 1.0
    ref = (Rotation.from_rotvec([0, 0.5, 0]) * Rotation.from_rotvec([0.5, 0, 0])).as_rotvec()
    assert np.allclose(_integrate_libero(dd)[0, 1, 3:6], ref)
    # consensus horizon counts orientation and gripper, excludes the chosen chunk
    cg = np.repeat(base[None], 20, 0); cg[10:, 4:, 6] = 0.0         # same path, 10 of 20 close the gripper at step 5
    assert consensus_horizon(cg, 0) == 4                            # 9/19 others agree after step 4 < 0.5
    cr = np.repeat(base[None], 4, 0); cr[1:, 6:, 3] += 1.0          # 3 others rotate by 1 rad (= 5 cm) from step 7
    assert consensus_horizon(cr, 0) == 6                            # self not counted: 0/3 agree, not 1/4
    # exact gripper ties are not voted on
    tie = np.repeat(base[None], 20, 0); tie[10:, 3, 6] = 0.0
    assert len(select_chunk(tie, 8).kept) == 20
    for bad in (np.full((2, 3, 7), np.nan), np.zeros((2, 3, 5))):
        try:
            select_chunk(bad, 2); raise AssertionError("invalid input accepted")
        except ValueError:
            pass
    try:
        select_egc(np.zeros((2, 3, 7)), 2, action_space="delta"); raise AssertionError("bad action_space accepted")
    except ValueError:
        pass
    print("[PASS] ccs: gripper vote drops minority, medoid avoids outlier, coherence follows previous plan, "
          "identical samples -> 0, euler wrap safe; same behaviour on LIBERO delta actions; "
          "EGC stays/switches by support; XDC = medoid without history, follows the previous distribution with it; "
          "consensus horizon stops where the majority leaves the chosen chunk; KeyStone clusters bimodal sets; "
          "rate limiter clips position only; review fixes: local EGC radius, SO(3) LIBERO metric, full-pose+gripper "
          "consensus without self, gripper ties, input validation")


if __name__ == "__main__":
    _selftest()
