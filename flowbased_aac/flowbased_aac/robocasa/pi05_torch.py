"""π0.5 (LeRobot PyTorch, ``lerobot/pi05_robocasa``) with AAC multi-sampling and DUAC divergence.

Preprocessing reproduces the checkpoint's LeRobot pipeline (policy_preprocessor.json):
  state  -> MEAN_STD normalization -> 256-bin discretization in the prompt
            "Task: {task}, State: {bins};\\nAction: " (Pi05PrepareStateTokenizerProcessorStep)
  prompt -> PaliGemma SentencePiece tokenizer with BOS, right padding to 200 tokens
            (the HF tokenizer google/paligemma-3b-pt-224 is gated; the SentencePiece model is the same vocabulary)
  images -> uint8 HWC 256x256 -> [0, 1] -> PI05Policy._preprocess_images (resize-pad 224, [-1, 1])
  action <- MEAN_STD un-normalization of the first 12 of 32 padded dims.

Flow convention (same as OpenPI): t = 1 is noise, x <- x + dt * v with dt = -1/steps, v = noise - action.
DUAC (paper time tau, increasing to data): after tau * steps Euler steps x = a_tau, model time t = 1 - tau.
The paper field is -v, so its Jacobian trace is minus the model's. Per action position i,
    U_i = (1 - tau)^2 / tau * (D + (1 - tau) * Tr(d v_paper,i / d a_tau,i))     (equation 6)
with D = 12 real action dims. The block trace is a Hutchinson estimate with Rademacher probes masked to the
12 real dims; it uses vector-Jacobian products: E[eps^T (J^T eps)]_i = Tr(J_ii), K probes in one batch.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np

ACTION_DIM = 12
STATE_DIM = 16
IMAGE_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_right",
]
CAMERAS = [k.rsplit(".", 1)[1] for k in IMAGE_KEYS]


def read_safetensors_f32(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        payload = handle.read()
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            raise ValueError(f"{key}: expected F32, got {meta['dtype']}")
        start, end = meta["data_offsets"]
        out[key] = np.frombuffer(payload[start:end], dtype=np.float32).reshape(meta["shape"]).copy()
    return out


class Pi05RoboCasa:
    def __init__(self, checkpoint: str | Path, tokenizer_model: str | Path, device: str = "cuda", seed: int = 0):
        import sentencepiece
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        checkpoint = Path(checkpoint)
        stats_file = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        stats = read_safetensors_f32(stats_file)
        for key in ("observation.state.mean", "observation.state.std", "action.mean", "action.std"):
            if key not in stats or not np.any(stats[key]):
                raise ValueError(f"{stats_file}: missing or all-zero {key} (corrupted copy?)")
        self.device = device
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.compile_model = False
        config.gradient_checkpointing = False
        config.device = device
        self.policy = PI05Policy.from_pretrained(checkpoint, config=config).to(device).eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.model = self.policy.model
        self.config = self.policy.config
        self.state_mean, self.state_std = stats["observation.state.mean"], stats["observation.state.std"]
        self.action_mean = torch.tensor(stats["action.mean"][:ACTION_DIM], device=device)
        self.action_std = torch.tensor(stats["action.std"][:ACTION_DIM], device=device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_model))
        self.max_length = int(self.config.tokenizer_max_length)
        self.steps = int(self.config.num_inference_steps)
        self.chunk = int(self.config.chunk_size)
        self.max_action_dim = int(self.config.max_action_dim)
        self.generator = torch.Generator(device=device).manual_seed(int(seed))  # stateful across calls

    # ------------------------------------------------------------------ inputs
    def prompt(self, task: str, state: np.ndarray) -> str:
        normalized = (np.asarray(state, dtype=np.float32) - self.state_mean) / (self.state_std + 1e-8)
        bins = np.digitize(normalized, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        return f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "

    def _tokens(self, prompt: str):
        import torch

        ids = self.tokenizer.encode(prompt, add_bos=True)[: self.max_length]
        tokens = np.zeros(self.max_length, dtype=np.int64)
        mask = np.zeros(self.max_length, dtype=bool)
        tokens[: len(ids)] = ids
        mask[: len(ids)] = True
        return torch.tensor(tokens, device=self.device)[None], torch.tensor(mask, device=self.device)[None]

    def _prefix(self, images: List[np.ndarray], tokens, masks):
        import torch
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d

        batch = {key: torch.tensor(np.ascontiguousarray(img), device=self.device).permute(2, 0, 1)[None].float() / 255.0
                 for key, img in zip(IMAGE_KEYS, images, strict=True)}
        imgs, img_masks = self.policy._preprocess_images(batch)
        emb, pad_masks, att_masks = self.model.embed_prefix(imgs, img_masks, tokens, masks)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        positions = torch.cumsum(pad_masks, dim=1) - 1
        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, cache = self.model.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(att_2d), position_ids=positions,
            past_key_values=None, inputs_embeds=[emb, None], use_cache=True)
        return pad_masks, cache

    @staticmethod
    def _expand(pad_masks, cache, n: int):
        from transformers import DynamicCache

        if n == 1:
            return pad_masks, cache
        layers = tuple((k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1), w) for k, v, w in cache)
        return pad_masks.repeat(n, 1), DynamicCache(layers)

    def _velocity(self, pad_masks, cache, x, t: float):
        import torch

        return self.model.denoise_step(pad_masks, cache, x, torch.full((x.shape[0],), t, device=self.device))

    def _unnormalize(self, x):
        return x[..., :ACTION_DIM] * self.action_std + self.action_mean

    # ------------------------------------------------------------------ AAC: N samples
    def sample(self, images, state, task: str, n: int = 1) -> Dict[str, Any]:
        import torch

        started = time.perf_counter()
        prompt = self.prompt(task, state)
        with torch.no_grad():
            pad_masks, cache = self._prefix(images, *self._tokens(prompt))
            pad_n, cache_n = self._expand(pad_masks, cache, n)
            x = torch.randn((n, self.chunk, self.max_action_dim), generator=self.generator, device=self.device)
            dt = -1.0 / self.steps
            for step in range(self.steps):
                x = x + dt * self._velocity(pad_n, cache_n, x, 1.0 + step * dt)
            actions = self._unnormalize(x)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return {"actions": actions.float().cpu().numpy(), "normalized_action": x[..., :ACTION_DIM].float().cpu().numpy(),
                "prompt": prompt, "infer_ms": 1000 * (time.perf_counter() - started)}

    # ------------------------------------------------------------------ DUAC: one chunk + divergence
    def sample_duac(self, images, state, task: str, tau: float = 0.9, num_probes: int = 40) -> Dict[str, Any]:
        import torch

        target = int(round(tau * self.steps))
        if not np.isclose(tau * self.steps, target):
            raise ValueError("tau * num_inference_steps must be an integer")
        started = time.perf_counter()
        prompt = self.prompt(task, state)
        dt = -1.0 / self.steps
        with torch.no_grad():
            pad_masks, cache = self._prefix(images, *self._tokens(prompt))
            x = torch.randn((1, self.chunk, self.max_action_dim), generator=self.generator, device=self.device)
            for step in range(target):
                x = x + dt * self._velocity(pad_masks, cache, x, 1.0 + step * dt)
            probes = torch.randint(0, 2, (num_probes, self.chunk, self.max_action_dim), generator=self.generator,
                                   device=self.device).float() * 2 - 1
            probes[..., ACTION_DIM:] = 0.0  # padded action dims carry no action
            pad_k, cache_k = self._expand(pad_masks, cache, num_probes)
        model_time = 1.0 - tau
        with torch.enable_grad():
            xk = x.repeat(num_probes, 1, 1).requires_grad_(True)
            v = self._velocity(pad_k, cache_k, xk, model_time)
            (vjp,) = torch.autograd.grad((v * probes).sum(), xk)  # J^T eps for every probe (samples independent)
        with torch.no_grad():
            model_div = (probes * vjp).sum(-1).mean(0)  # (chunk,) estimate of Tr(dv_i/dx_i), model time
            paper_div = -model_div
            scale = (1.0 - tau) ** 2 / tau
            uncertainty = scale * (ACTION_DIM + (1.0 - tau) * paper_div)
            x = x + dt * v[:1].detach()  # Euler step at tau reuses the velocity already computed
            for step in range(target + 1, self.steps):
                x = x + dt * self._velocity(pad_masks, cache, x, 1.0 + step * dt)
            actions = self._unnormalize(x)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return {"actions": actions[0].float().cpu().numpy(), "normalized_action": x[0, :, :ACTION_DIM].float().cpu().numpy(),
                "u_step": uncertainty.float().cpu().numpy(), "duac_divergence": paper_div.float().cpu().numpy(),
                "prompt": prompt, "infer_ms": 1000 * (time.perf_counter() - started)}
