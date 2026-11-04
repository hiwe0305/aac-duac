"""CPU-only stand-ins for LIBERO and the policy servers, used by ``--synthetic-dry-run`` and tests.

They expose exactly the attributes the real evaluation loop touches, so a dry run exercises the
same code path (decision rule, logging, subgoals, state bank, resume) without MuJoCo or a GPU.
Success rates from a dry run are meaningless.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _Task(SimpleNamespace):
    pass


class _TaskSuite:
    def __init__(self, suite: str, n_tasks: int = 10, n_init: int = 50):
        self.n_tasks = n_tasks
        self._n_init = n_init
        self._suite = suite

    def get_task(self, i):
        return _Task(name=f"synthetic_task_{i}", language=f"synthetic task {i}", problem_folder=self._suite,
                     bddl_file=f"task_{i}.bddl")


class _Env:
    """Two goal predicates that become true after ~1/3 and ~4/5 of a 120-step episode."""

    def __init__(self, task_id: int):
        self._task_id = task_id
        self._t = 0
        self._state = np.zeros(8)
        self.robots = [SimpleNamespace(controller=SimpleNamespace(initial_joint=np.zeros(7)),
                                       gripper=SimpleNamespace(current_action=np.zeros(1)))]
        self.sim = SimpleNamespace(data=SimpleNamespace(qacc_warmstart=np.zeros(9)))
        self.env = self
        self.parsed_problem = {"goal_state": [["turnon", "stove"], ["on", "moka_pot", "stove_region"]]}

    def _obs(self):
        return {
            "agentview_image": np.zeros((16, 16, 3), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((16, 16, 3), np.uint8),
            "robot0_eef_pos": self._state[:3].copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def reset(self):
        self._t = 0
        return self._obs()

    def set_init_state(self, state):
        self._state = np.asarray(state, dtype=np.float64).copy()
        self._t = 0
        return self._obs()

    def get_sim_state(self):
        return np.r_[self._t, self._state]

    def _eval_predicate(self, predicate):
        limit = 40 if predicate[0] == "turnon" else 100 + 10 * (self._task_id % 3)
        return self._t >= limit

    def step(self, action):
        self._t += 1
        self._state[:3] += np.asarray(action[:3]) * 0.01
        self.robots[0].gripper.current_action = np.asarray(action[6:7])
        done = all(self._eval_predicate(g) for g in self.parsed_problem["goal_state"])
        return self._obs(), float(done), done, {}

    def close(self):
        pass


class _Client:
    def __init__(self, multi_sample: bool, n_samples: int = 20, horizon: int = 10, seed: int = 0):
        self._multi = multi_sample
        self._n = n_samples
        self._h = horizon
        self._rng = np.random.default_rng(seed)

    def infer(self, _request):
        mean = np.tile(np.array([0.4, 0.2, -0.1, 0.02, -0.01, 0.01, -1.0]), (self._h, 1))
        if self._multi:
            spread = np.linspace(0.02, 0.3, self._h)[None, :, None]
            actions = mean[None] + self._rng.normal(size=(self._n, self._h, 7)) * spread
            actions[..., 6] = np.where(self._rng.random((self._n, self._h)) < 0.1, 1.0, -1.0)
            normalized = np.concatenate([actions, np.zeros((self._n, self._h, 25))], axis=-1)
            return {"actions": actions, "normalized_action": normalized, "policy_timing": {"infer_ms": 1.0}}
        actions = mean + self._rng.normal(size=(self._h, 7)) * 0.05
        u_step = 0.01 + 0.002 * np.arange(self._h) + np.where(np.arange(self._h) >= self._rng.integers(3, 9), 0.05, 0.0)
        normalized = np.concatenate([actions, np.zeros((self._h, 25))], axis=-1)
        return {"actions": actions, "normalized_action": normalized, "u_step": u_step,
                "policy_timing": {"infer_ms": 1.0}}


class SyntheticBackend:
    def __init__(self, suite: str, multi_sample: bool):
        self.suite = suite
        self.task_suite = _TaskSuite(suite)
        self.client = _Client(multi_sample)

    def init_states(self, task_id: int):
        return [np.full(8, 0.01 * k) for k in range(self.task_suite._n_init)]

    def make_env(self, task, seed: int):
        return _Env(int(task.name.rsplit("_", 1)[1])), task.language
