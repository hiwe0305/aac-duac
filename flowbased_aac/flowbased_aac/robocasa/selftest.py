#!/usr/bin/env python3
"""Checks of the RoboCasa π0.5 DUAC/AAC sampler that need no checkpoint or GPU.

The model's velocity field is replaced by an affine map v(x) = A x + b with known Jacobian, so equation (6)
can be compared with its exact value: U_i = (1-tau)^2/tau * (12 + (1-tau) * (-Tr A_ii)), where A_ii is the
12x12 block of action position i (padded dims 12..31 must not contribute).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    import torch

    from flowbased_aac.robocasa.pi05_torch import ACTION_DIM, Pi05RoboCasa

    torch.manual_seed(0)
    chunk, dim, tau = 50, 32, 0.9
    m = Pi05RoboCasa.__new__(Pi05RoboCasa)  # skip checkpoint loading
    m.device, m.steps, m.chunk, m.max_action_dim = "cpu", 10, chunk, dim
    m.generator = torch.Generator().manual_seed(0)
    m.action_mean, m.action_std = torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)
    A = torch.randn(chunk * dim, chunk * dim) * 0.05
    b = torch.randn(chunk * dim) * 0.1
    m.prompt = lambda task, state: "p"
    m._tokens = lambda prompt: (None, None)
    m._prefix = lambda images, t, mk: (torch.zeros(1, 4), None)
    m._expand = lambda pad, cache, n: (pad.repeat(n, 1), None)
    m._velocity = lambda pad, cache, x, t: (x.reshape(x.shape[0], -1) @ A.T + b).reshape(x.shape)

    out = m.sample_duac(None, None, "t", tau=tau, num_probes=20000)
    blocks = A.reshape(chunk, dim, chunk, dim)
    tr = torch.stack([torch.trace(blocks[i, :ACTION_DIM, i, :ACTION_DIM]) for i in range(chunk)]).numpy()
    exact = (1 - tau) ** 2 / tau * (ACTION_DIM + (1 - tau) * (-tr))
    err = np.abs(out["u_step"] - exact).max()
    rel = err / ((1 - tau) ** 3 / tau)
    assert np.allclose(out["duac_divergence"], -tr, atol=0.15), (out["duac_divergence"][:5], -tr[:5])
    print(f"[PASS] DUAC eq.6 vs exact block traces: max |U - U*| = {err:.2e} ({rel:.3f} in trace units); sign and padding mask OK")

    # The final chunk must equal plain Euler integration with the same noise (the tau step reuses its velocity)
    m.generator = torch.Generator().manual_seed(1)
    noise = torch.randn((1, chunk, dim), generator=torch.Generator().manual_seed(1))
    x = noise.clone()
    for s in range(10):
        x = x + (-0.1) * m._velocity(None, None, x, 1 - 0.1 * s)
    out = m.sample_duac(None, None, "t", tau=tau, num_probes=4)
    assert np.allclose(out["actions"], x[0, :, :ACTION_DIM].numpy(), atol=1e-4)
    print("[PASS] DUAC chunk equals plain 10-step Euler integration of the same noise")

    # AAC: N independent samples, shapes (N, 50, 12), different noise per sample and per call (stateful RNG)
    m.generator = torch.Generator().manual_seed(0)
    s1, s2 = m.sample(None, None, "t", n=20), m.sample(None, None, "t", n=20)
    assert s1["actions"].shape == (20, chunk, ACTION_DIM)
    assert not np.allclose(s1["actions"][0], s1["actions"][1]) and not np.allclose(s1["actions"], s2["actions"])
    print("[PASS] AAC sampler: (20, 50, 12), distinct noise per sample and per call")
    print("[robocasa selftest] ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
