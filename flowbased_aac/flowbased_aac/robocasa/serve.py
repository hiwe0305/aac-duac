#!/usr/bin/env python3
"""Websocket server for π0.5 RoboCasa (OpenPI msgpack protocol, usable with openpi_client).

  --mode duac : one chunk + DUAC uncertainty (equation 6) and divergence per action position.
                Also used by the Fixed-h and Random-h arms, so DUAC's proposal is always logged.
  --mode aac  : N sampled chunks (AAC), sharing one prefix encoding.

Request : {"images": [3 x (256, 256, 3) uint8 in CAMERAS order], "state": (16,), "task": str}
Response: duac -> actions (50, 12), normalized_action (50, 12), u_step (50,), duac_divergence (50,), duac_horizon
          aac  -> actions (N, 50, 12), normalized_action (N, 50, 12)
          both -> policy_timing.infer_ms, prompt
The RNG is a single stateful torch.Generator (seeded once), as in the O-LONG protocol.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import threading
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "flowbased_aac", ROOT / "external" / "openpi" / "packages" / "openpi-client" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flowbased_aac.duac_policy import prefix_mean_jump_horizon  # noqa: E402
from flowbased_aac.robocasa.pi05_torch import Pi05RoboCasa  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["duac", "aac"], required=True)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "lerobot" / "pi05_robocasa"))
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints" / "openpi" / "big_vision" / "paligemma_tokenizer.model"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--num-probes", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=20)
    args = ap.parse_args(argv)

    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    policy = Pi05RoboCasa(args.checkpoint, args.tokenizer, device=args.device, seed=args.seed)
    lock = threading.Lock()
    metadata = {"mode": args.mode, "chunk_size": policy.chunk, "action_dim": 12, "seed": args.seed,
                "tau": args.tau, "num_probes": args.num_probes, "n_samples": args.n_samples,
                "checkpoint": str(Path(args.checkpoint).resolve())}

    def infer(obs: dict) -> dict:
        images = [np.asarray(im) for im in obs["images"]]
        state = np.asarray(obs["state"], dtype=np.float32)
        with lock:
            if args.mode == "duac":
                out = policy.sample_duac(images, state, str(obs["task"]), tau=args.tau, num_probes=args.num_probes)
                out["duac_horizon"] = np.asarray(prefix_mean_jump_horizon(out["u_step"], 2)[0], dtype=np.int32)
            else:
                out = policy.sample(images, state, str(obs["task"]), n=args.n_samples)
        out["policy_timing"] = {"infer_ms": out.pop("infer_ms")}
        return out

    def handler(ws):
        packer = msgpack_numpy.Packer()
        ws.send(packer.pack(metadata))
        while True:
            try:
                msg = ws.recv()
            except Exception:
                return
            try:
                ws.send(packer.pack(infer(msgpack_numpy.unpackb(msg))))
            except Exception:
                ws.send(traceback.format_exc())
                raise

    with serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        print(f"[robocasa serve] mode={args.mode} on {args.host}:{args.port} server listening", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
