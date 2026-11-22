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
               gamma: float = 0.5, rot_weight: float = 0.05, action_space: str = "absolute") -> EGCResult:
    """Evidence-Gated Commitment: stay in the previous plan's mode while the fresh samples still support it.

    The previous plan is represented by its unexecuted tail, which covers the first L = H - h_prev steps of the new
    chunk (same time steps). On that overlap window:
      r_L      = median pairwise overlap distance among the N samples
      peak     = max_j #{k : d_L(a_j, a_k) <= r_L}              mass of the best-supported mode
      support  = #{k : d_L(a_k, tail) <= r_L}                    mass that still agrees with the previous plan
      commit  <=> support >= gamma * peak
      commit -> medoid (distance on the executed prefix h) of the continuation set {k : d_L(a_k, tail) <= r_L}
      switch -> medoid of all N samples
    With no previous chunk or no overlap the global medoid is executed. Unlike a soft coherence cost (BID backward
    coherence, CCS) the previous plan is never followed against the evidence: once fewer than gamma x peak samples
    agree with it, the policy switches; unlike the memoryless medoid it does not flip between two comparably supported
    modes at every replan.
    """

    samples = np.asarray(samples, dtype=np.float64)[..., :7]
    n, horizon, _ = samples.shape
    h = int(max(1, min(h, horizon)))
    if action_space == "libero_delta":
        rot_mode = "vector"
        traj = _integrate_libero(samples)
        if prev_chunk is not None:
            prev_chunk = np.asarray(prev_chunk, dtype=np.float64)[:, :7]
            prev_chunk = np.concatenate([np.zeros((prev_h, 7)), _integrate_libero(prev_chunk[prev_h:])], axis=0)
    else:
        rot_mode = "euler"
        traj = samples
        if prev_chunk is not None:
            prev_chunk = np.asarray(prev_chunk, dtype=np.float64)[:, :7]
    pre = traj[:, :h]
    dfull = _pose_distance(pre[:, None], pre[None, :], rot_weight, rot_mode)
    medoid = int(np.argmin(dfull.sum(axis=1)))
    L = int(max(0, min(horizon, prev_chunk.shape[0] - prev_h))) if prev_chunk is not None else 0
    if L == 0:
        return EGCResult(medoid, False, 0, 0, 0, 0, 0.0)
    win = traj[:, :L]
    dL = _pose_distance(win[:, None], win[None, :], rot_weight, rot_mode)
    r = float(np.median(dL[~np.eye(n, dtype=bool)]))
    peak = int((dL <= r).sum(axis=1).max())
    dtail = _pose_distance(win, prev_chunk[prev_h:prev_h + L][None], rot_weight, rot_mode)
    cont = np.flatnonzero(dtail <= r)
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
    A product of the two sample-based densities across time: transient, observation-driven flips of the policy's
    distribution are damped by the previous distribution, while a persistent change is followed one decision later
    (the previous distribution has moved by then). Unlike coherence with the single executed chunk (BID, CCS) it does
    not lock onto one possibly atypical chunk. Without previous samples it is the medoid.
    """

    samples = np.asarray(samples, dtype=np.float64)[..., :7]
    n, horizon, _ = samples.shape
    h = int(max(1, min(h, horizon)))
    delta = action_space == "libero_delta"
    rot_mode = "vector" if delta else "euler"
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


def consensus_horizon(samples: np.ndarray, index: int, radius: float = 0.02, quorum: float = 0.5,
                      min_h: int = 2, action_space: str = "absolute") -> int:
    """How many steps of the chosen chunk the samples still agree on (uncertainty of the executed chunk itself).

    agree_k = fraction of the N samples whose end-effector position at step k is within ``radius`` (m) of the chosen
    chunk's; k* = the last step before agree_k first drops below ``quorum``; returns max(min_h, k*). Unlike AAC's
    marginal entropy over all samples or DUAC's divergence of one random sample, it measures agreement with the chunk
    that will actually be executed.
    """

    samples = np.asarray(samples, dtype=np.float64)[..., :7]
    traj = _integrate_libero(samples) if action_space == "libero_delta" else samples
    d = np.linalg.norm(traj[:, :, :3] - traj[index][None, :, :3], axis=-1)  # (N, H)
    low = (d <= radius).mean(axis=0) < quorum
    k = int(np.argmax(low)) if low.any() else samples.shape[1]
    return int(max(min_h, min(k, samples.shape[1])))


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
    # EGC: previous plan followed R (its unexecuted tail sits on R's first steps)
    prevR = np.concatenate([np.zeros((8, 7)), R[:2]], axis=0); prevR[:8] = R[:8]
    e1 = select_egc(fresh, 8, prevR, 8)                    # R has 8 vs L 12 samples: 8 >= 0.5 * 12 -> stay in R
    assert e1.commit and e1.index >= 12, e1
    e2 = select_egc(fresh, 8, prevR, 8, gamma=0.9)         # stricter gate -> switch to the global medoid (L)
    assert not e2.commit and e2.index < 12, e2
    allL = L[None] + rng2.normal(0, 0.002, (20, 10, 7)) * np.r_[1, 1, 1, 0, 0, 0, 0]
    e3 = select_egc(allL, 8, prevR, 8)                     # R lost all support -> switch
    assert not e3.commit and e3.support < 0.5 * e3.peak, e3
    assert select_egc(fresh, 8).commit is False            # first decision -> medoid
    dR = np.diff(np.concatenate([np.zeros((20, 1, 7)), fresh], axis=1), axis=1); dR[..., :3] /= LIBERO_POS_UNIT
    dR[..., 3:6] = 0; dR[..., 6] = -1.0
    pR = np.zeros((10, 7))  # executed 8 steps (irrelevant), then a tail that integrates to R[:2] from the current pose
    pR[8:] = np.diff(np.concatenate([np.zeros((1, 7)), R[:2]], axis=0), axis=0); pR[:, :3] /= LIBERO_POS_UNIT
    pR[:, 3:6] = 0; pR[:, 6] = -1.0
    e4 = select_egc(dR, 8, pR, 8, action_space="libero_delta")
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
    print("[PASS] ccs: gripper vote drops minority, medoid avoids outlier, coherence follows previous plan, "
          "identical samples -> 0, euler wrap safe; same behaviour on LIBERO delta actions; "
          "EGC stays/switches by support; XDC = medoid without history, follows the previous distribution with it; "
          "consensus horizon stops where the majority leaves the chosen chunk")


if __name__ == "__main__":
    _selftest()
