#!/usr/bin/env python3
"""VLABench adapter checks that need no simulator, checkpoint or GPU (run with .venv-vlabench or .venv-libero).

  1. Conventions: euler <-> quaternion equals VLABench's utils; policy_state / to_env_target invert each other.
  2. to_aac_frame: recovers known per-step deltas (pos + world-frame rotation), composing the rotation deltas with the
     official AAC compose_rotations gives the rotation from the current pose to target h, gripper/units mapping.
  3. AAC / DUAC are called unchanged: the official select_chunk_size on the converted chunk equals aac_decision, and
     select_duac_horizon gives max(eq. 8, xi) on it; xi binds in LIBERO units and (practically) never in raw units.
  4. The real evaluation loop (eval_vlabench.run) on a synthetic env + server for every arm, resume, and errors.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "flowbased_aac"))


def main() -> int:
    from scipy.spatial.transform import Rotation

    from flowbased_aac.aac_official import _official, aac_decision
    from flowbased_aac.duac_policy import DUACConfig, compose_rotations, prefix_mean_jump_horizon, select_duac_horizon
    from flowbased_aac.vlabench import common

    rng = np.random.default_rng(0)
    # 1. conventions --------------------------------------------------------------------------------------------
    for _ in range(100):
        e = rng.uniform([-np.pi, -1.4, -np.pi], [np.pi, 1.4, np.pi])
        q = common.euler_xyz_to_quat_wxyz(e)
        ref = Rotation.from_euler("xyz", e).as_quat()  # x y z w
        assert np.allclose(np.abs(np.dot(q[[1, 2, 3, 0]], ref)), 1.0, atol=1e-9)
        assert np.allclose(Rotation.from_euler("xyz", common.quat_wxyz_to_euler_xyz(q)).as_matrix(),
                           Rotation.from_euler("xyz", e).as_matrix(), atol=1e-9)
    ee = np.r_[0.1, -0.2, 1.1, common.euler_xyz_to_quat_wxyz([3.1, -0.05, -1.6]), 1.0]
    s = common.policy_state(ee)
    pos, quat, fingers = common.to_env_target(s)
    assert np.allclose(pos, ee[:3]) and np.allclose(np.abs(np.dot(quat, ee[3:7])), 1.0) and np.allclose(fingers, 0.04)
    assert np.allclose(common.to_env_target(np.r_[s[:6], 0.05])[2], 0.0)
    print("[PASS] euler/quat conventions match VLABench; state <-> env target round trip; gripper threshold 0.1")

    # 2. to_aac_frame -------------------------------------------------------------------------------------------
    H = 10
    state = np.r_[0.0, 0.2, 0.4, 3.12, -0.05, -1.6, 1.0]
    dpos = rng.normal(0, 0.01, (H, 3))
    drot = rng.normal(0, 0.05, (H, 3))
    rot = Rotation.from_euler("xyz", state[3:6])
    chunk = np.zeros((H, 7))
    p = state[:3].copy()
    for k in range(H):
        p = p + dpos[k]
        rot = Rotation.from_rotvec(drot[k]) * rot  # world-frame delta, as LIBERO / robosuite OSC
        chunk[k, :3], chunk[k, 3:6] = p, rot.as_euler("xyz")  # euler wraps at +-pi: handled via rotations
    chunk[:, 6] = np.r_[np.ones(6), np.zeros(4)]  # open ... then close
    raw = common.to_aac_frame(chunk, state, units="raw")
    assert np.allclose(raw[:, :3], dpos, atol=1e-12) and np.allclose(raw[:, 3:6], drot, atol=1e-9)
    assert np.array_equal(raw[:, 6], np.r_[-np.ones(6), np.ones(4)])
    for h in (1, 4, 10):
        total = Rotation.from_euler("xyz", chunk[h - 1, 3:6]) * Rotation.from_euler("xyz", state[3:6]).inv()
        assert np.isclose(np.linalg.norm(compose_rotations(raw[:h, 3:6])), total.magnitude(), atol=1e-9)
    lib = common.to_aac_frame(chunk, state)
    assert np.allclose(lib[:, :3], dpos / 0.05) and np.allclose(lib[:, 3:6], drot / 0.5) and np.array_equal(lib[:, 6], raw[:, 6])
    batch = common.to_aac_frame(np.stack([chunk, chunk]), state)
    assert batch.shape == (2, H, 7) and np.allclose(batch[1], lib)
    print("[PASS] to_aac_frame: per-step deltas recovered (euler wrap safe), composed rotation = pose change, "
          "LIBERO units 0.05 m / 0.5 rad, gripper open -> -1 / close -> +1, batched")

    # 3. AAC / DUAC unchanged on the converted chunk -------------------------------------------------------------
    aac = _official()
    for trial in range(50):
        samples = chunk[None] + rng.normal(0, 1, (20, H, 7)) * np.linspace(0.001, 0.03, H)[None, :, None]
        samples[..., 6] = np.where(rng.random((20, H)) < 0.1 + 0.08 * np.arange(H), 0.0, 1.0)
        conv = common.to_aac_frame(samples, state)
        norm = np.pad(samples - state, ((0, 0), (0, 0), (0, 25)))
        d = aac_decision(conv, norm, selector="gaussian_bernoulli", move_th=3.0)
        pred = {"normalized_action": norm, "action.end_effector_position": conv[..., :3],
                "action.end_effector_rotation": conv[..., 3:6], "action.gripper_close": conv[..., 6]}
        official_h, _ = aac.select_chunk_size(pred, method="gaussian_bernoulli", move_th=3.0)
        assert d["h_selected"] == int(official_h) and 2 <= d["h_selected"] <= H
        u = rng.random(H)
        res = select_duac_horizon(u, conv[0], DUACConfig(chunk_horizon=H))
        assert res.h_selected == min(H, max(prefix_mean_jump_horizon(u, 2)[0], res.xi))
    fast = chunk.copy()
    fast[:, :3] = state[:3] + np.cumsum(np.tile([0.03, 0.0, 0.0], (H, 1)), axis=0)  # 3 cm / step
    xi_lib = select_duac_horizon(np.zeros(H), common.to_aac_frame(fast, state), DUACConfig(chunk_horizon=H)).xi
    xi_raw = select_duac_horizon(np.zeros(H), common.to_aac_frame(fast, state, units="raw"), DUACConfig(chunk_horizon=H)).xi
    assert xi_lib < H and xi_raw == H, (xi_lib, xi_raw)
    print(f"[PASS] official AAC select_chunk_size == aac_decision on converted chunks (50 draws); DUAC h = max(eq.8, xi); "
          f"xi at 3 cm/step: {xi_lib} (LIBERO units) vs {xi_raw} (raw metres)")

    # 4. real loop, synthetic env/server ------------------------------------------------------------------------
    from flowbased_aac.vlabench import eval_vlabench as ev

    tmp = Path(tempfile.mkdtemp(prefix="vlabench_selftest_"))
    try:
        base = ["--synthetic-dry-run", "--track", "T1", "--tasks", "select_fruit,select_toy", "--episodes", "2"]
        runs = {"duac": [], "fixed_4": ["--fixed-horizon", "4"], "vanilla": ["--vanilla"],
                "aac": ["--aac-selector", "gaussian_bernoulli"]}
        for name, extra in runs.items():
            summary = ev.run(ev.build_parser().parse_args(base + extra + ["--state-bank", "--output-dir", str(tmp / name)]))
            assert summary["arm"] == name and summary["episodes"] == 4 and summary["errors"] == 0, summary
            dec = ev.read_rows(tmp / name / "decisions.csv")
            hs = {int(r["h_executed"]) for r in dec}
            assert all(1 <= h <= 10 for h in hs)
            if name == "fixed_4":
                assert hs == {4}
            if name == "vanilla":
                assert hs == {5}
            steps = [int(r["sim_step"]) for r in dec if r["task"] == "select_fruit" and r["episode"] == "0"]
            assert steps == sorted(steps) and steps[0] == 0
            eps = ev.read_rows(tmp / name / "episodes.csv")
            assert all(int(r["end_step"]) <= int(r["max_steps"]) for r in eps)
            assert len(list((tmp / name / "state_bank").glob("*.npz"))) == 4
        for name, extra in (("duac_noxi", ["--no-xi"]), ("aac_noxi", ["--no-xi", "--aac-selector", "gaussian_bernoulli"])):
            summary = ev.run(ev.build_parser().parse_args(base + extra + ["--output-dir", str(tmp / name)]))
            dec = ev.read_rows(tmp / name / "decisions.csv")
            assert summary["arm"] == name and dec and all(int(r["h_executed"]) == int(r["h_unc"]) for r in dec), name
            assert all(int(r["h_method"]) == min(int(r["prediction_horizon"]), max(int(r["h_unc"]), int(r["move_lower_bound"]))) for r in dec)
        try:
            ev.run(ev.build_parser().parse_args(base + ["--no-xi", "--fixed-horizon", "4", "--output-dir", str(tmp / "bad2")]))
            raise AssertionError("--no-xi with a fixed arm must fail")
        except ValueError:
            pass
        print("[PASS] ablation --no-xi: duac_noxi / aac_noxi execute h = h_unc, full proposal max(h_unc, xi) still logged")
        summary = ev.run(ev.build_parser().parse_args(base + ["--random-horizon-from", str(tmp / "duac" / "decisions.csv"),
                                                              "--output-dir", str(tmp / "random_h")]))
        assert summary["arm"] == "random_h"
        # resume: delete the last episode row, rerun -> exactly one episode re-evaluated, no duplicate decisions
        rows = (tmp / "duac" / "episodes.csv").read_text().splitlines()
        (tmp / "duac" / "episodes.csv").write_text("\n".join(rows[:-1]) + "\n")
        before = len(ev.read_rows(tmp / "duac" / "decisions.csv"))
        summary = ev.run(ev.build_parser().parse_args(base + ["--state-bank", "--resume-dir", str(tmp / "duac")]))
        assert summary["episodes"] == 4
        after = ev.read_rows(tmp / "duac" / "decisions.csv")
        keys = [(r["task"], r["episode"], r["decision_id"]) for r in after]
        assert len(keys) == len(set(keys)) and len(after) == before, (len(after), before)
        try:
            ev.run(ev.build_parser().parse_args(base + ["--fixed-horizon", "3", "--resume-dir", str(tmp / "duac")]))
            raise AssertionError("resume with different args must fail")
        except ValueError:
            pass
        try:
            ev.run(ev.build_parser().parse_args(base + ["--fixed-horizon", "11", "--output-dir", str(tmp / "bad")]))
            raise AssertionError("h > prediction horizon must fail")
        except ValueError:
            pass
        # server failure mid-episode: no episode row, job raises; resume redoes the episode without duplicates
        calls = {"n": 0}
        orig = ev.SyntheticClient.infer
        def flaky(self, req):
            calls["n"] += 1
            if calls["n"] == 12:
                raise ConnectionError("keepalive ping timeout")
            return orig(self, req)
        ev.SyntheticClient.infer = flaky
        try:
            ev.run(ev.build_parser().parse_args(base + ["--output-dir", str(tmp / "flaky")]))
            raise AssertionError("server failure must abort the job")
        except ev.ServerError:
            pass
        finally:
            ev.SyntheticClient.infer = orig
        n_before = len(ev.read_rows(tmp / "flaky" / "episodes.csv"))
        assert all(not r["error"] for r in ev.read_rows(tmp / "flaky" / "episodes.csv"))
        summary = ev.run(ev.build_parser().parse_args(base + ["--resume-dir", str(tmp / "flaky")]))
        dec = ev.read_rows(tmp / "flaky" / "decisions.csv")
        keys = [(r["task"], r["episode"], r["decision_id"]) for r in dec]
        assert summary["episodes"] == 4 and summary["errors"] == 0 and n_before < 4 and len(keys) == len(set(keys))
        print("[PASS] evaluation loop (synthetic env/server): duac, fixed_4, vanilla (h=5), aac, random_h; state bank; "
              "resume without duplicates; arg mismatch and h > H rejected; server failure aborts without recording, resume redoes it")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[vlabench selftest] ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
