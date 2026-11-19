"""OpenPI train config ``pi05_ft_vlabench_primitive`` for the upstream OpenPI in external/openpi (.venv-openpi).

Reproduced from the official fork used to train ``VLABench/pi05-primitive-10task`` (Shiduo-zh/openpi, branch pi05,
cloned at external/openpi-vlabench, commit 788ef1f):
  * ``src/openpi/training/config.py``: TrainConfig ``pi05_ft_vlabench_primitive`` and ``LeRobotVLABenchDataConfig``
  * ``src/openpi/policies/vlabench_policy.py``: ``VLABenchInputs`` / ``VLABenchOutputs``
The fork differs from upstream OpenPI only in training code (a stop-gradient on the prefix, dataset options);
``Pi0.sample_actions``, ``ModelTransformFactory``, ``DataConfigFactory`` and the transforms used here are identical
(checked in docs/VLABENCH_INTEGRATION.md), so the checkpoint runs on the upstream OpenPI used by the LIBERO track.
Only inference-relevant fields are set; the dataset is never opened (norm stats come from the checkpoint assets).
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

CONFIG_NAME = "pi05_ft_vlabench_primitive"
ASSET_ID = "vlabench/vlabench_ft_primitive"  # checkpoint/assets/<ASSET_ID>/norm_stats.json


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_transform_classes():
    from openpi import transforms
    from openpi.models import model as _model

    @dataclasses.dataclass(frozen=True)
    class VLABenchInputs(transforms.DataTransformFn):
        action_dim: int
        model_type: _model.ModelType = _model.ModelType.PI0

        def __call__(self, data: dict) -> dict:
            inputs = {
                "state": transforms.pad_to_dim(data["observation/state"], self.action_dim),
                "image": {
                    "base_0_rgb": _parse_image(data["observation/image"]),
                    "left_wrist_0_rgb": _parse_image(data["observation/second_image"]),
                    "right_wrist_0_rgb": _parse_image(data["observation/wrist_image"]),
                },
                "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
            }
            if "actions" in data:
                inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
            if "prompt" in data:
                inputs["prompt"] = data["prompt"]
            return inputs

    @dataclasses.dataclass(frozen=True)
    class VLABenchOutputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"][:, :7])}

    return VLABenchInputs, VLABenchOutputs


def get_train_config():
    """TrainConfig equivalent to the fork's ``pi05_ft_vlabench_primitive`` (inference-relevant fields)."""

    import pathlib

    from openpi import transforms as _transforms
    from openpi.models import model as _model
    from openpi.models import pi0_config
    from openpi.training import config as _config

    VLABenchInputs, VLABenchOutputs = make_transform_classes()

    @dataclasses.dataclass(frozen=True)
    class LeRobotVLABenchDataConfig(_config.DataConfigFactory):
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
            repack = _transforms.Group(inputs=[_transforms.RepackTransform({
                "observation/image": "image", "observation/second_image": "second_image",
                "observation/wrist_image": "wrist_image", "observation/state": "state",
                "actions": "actions", "prompt": "prompt"})])
            data_transforms = _transforms.Group(
                inputs=[VLABenchInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
                outputs=[VLABenchOutputs()],
            )
            delta_action_mask = _transforms.make_bool_mask(6, -1)  # delta on xyz + euler, absolute gripper
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
            model_transforms = _config.ModelTransformFactory()(model_config)
            return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack,
                                       data_transforms=data_transforms, model_transforms=model_transforms)

    return _config.TrainConfig(
        name=CONFIG_NAME,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b"),
        data=LeRobotVLABenchDataConfig(repo_id=ASSET_ID, base_config=_config.DataConfig(prompt_from_task=True)),
    )
