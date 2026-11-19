# Task: Integrate VLABench into the Existing AAC and DUAC Evaluation Framework

## 1. Role and Project Context

You are a Senior AI Research Engineer specializing in Embodied AI, Vision-Language-Action (VLA) models, Flow Matching, Uncertainty Quantification, and Adaptive Action Chunking.

My existing research project already contains working implementations of:

- Adaptive Action Chunking (AAC).
- Adaptive Action Horizon Selection via Flow-Field Uncertainty (DUAC).
- Vanilla VLA inference.
- Fixed action chunking.
- Policy inference and evaluation pipelines.
- LIBERO benchmark integration.
- RoboCasa benchmark integration.
- Experimental logging and evaluation mechanisms.

AAC and DUAC have already been implemented and evaluated on LIBERO and RoboCasa.

Your objective is to extend the existing project to support VLABench as an additional evaluation benchmark.

This is an inference-time, training-free research task.

Do not train, fine-tune, or modify the pretrained VLA weights.
 
---

## 2. Research Objective

The primary objective is to evaluate the generalization, effectiveness, robustness, and computational efficiency of existing adaptive action chunking methods across different robotic manipulation benchmarks.

We want to compare:

- Vanilla π₀.₅.
- Fixed Action Chunking.
- π₀.₅ + AAC.
- π₀.₅ + DUAC.

The new benchmark must allow direct comparisons with our existing LIBERO and RoboCasa experiments.

The evaluation should investigate:

- Task success rate.
- Action-level uncertainty.
- Adaptive execution horizon.
- Inference efficiency.
- Computational overhead.
- Robustness under distribution shifts.
- Uncertainty behavior across different manipulation skills.

Do not claim that VLABench Primitive Tasks evaluate high-level long-horizon planning or inter-subgoal uncertainty propagation.

The scope is action-level uncertainty and adaptive execution.

---

## 3. Official Resources

Investigate the following official resources before implementing anything.

### VLABench

- Project: https://vlabench.github.io/
- GitHub: https://github.com/OpenMOSS/VLABench
- Paper: https://arxiv.org/abs/2412.18194
- Hugging Face: https://huggingface.co/VLABench
- LeRobot integration: https://huggingface.co/docs/lerobot/vlabench

### Target Checkpoint

- https://huggingface.co/VLABench/pi05-primitive-10task

### Reference Dataset

- https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot_video

### OpenPI

- https://github.com/Physical-Intelligence/openpi

### AAC

- Paper: https://arxiv.org/abs/2604.04161
- LIBERO implementation: https://github.com/Adaptive-Action-Chunking/libero
- RoboCasa implementation: https://github.com/Adaptive-Action-Chunking/robocasa

### DUAC

- https://openreview.net/forum?id=ryDGxkHES2

Use the existing AAC and DUAC implementations in my project as the primary source of truth.

External implementations should only be used to understand interfaces, verify correctness, and resolve compatibility issues.

---

## 4. Phase 1 — Existing Codebase Investigation

Before writing any new code, thoroughly inspect the current repository.

Identify:

- Project architecture.
- Existing benchmark adapters.
- LIBERO evaluation pipeline.
- RoboCasa evaluation pipeline.
- π₀.₅ model loading mechanism.
- Observation preprocessing.
- Action postprocessing.
- Flow Matching action generation.
- Action normalization and denormalization.
- AAC implementation.
- DUAC implementation.
- Adaptive horizon selection logic.
- Uncertainty estimation algorithms.
- Configuration system.
- Experiment logging.
- Evaluation metrics.
- Dependency management.
- Existing unit tests and integration tests.

Determine which components can be directly reused and which components require a VLABench-specific adapter.

Trace the complete existing inference pipeline:

```text
Environment Observation
    -> Observation Preprocessing
    -> VLA Policy
    -> Action Generation
    -> Uncertainty Estimation
    -> Adaptive Horizon Selection
    -> Action Execution
    -> Environment Feedback
```

Create a concise implementation plan based on the actual repository structure.

Do not assume that filenames, interfaces, or directory structures already exist.

Do not create an independent research project if the current architecture can be extended.

---

## 5. Phase 2 — VLABench Investigation

Investigate the official VLABench evaluation pipeline and determine:

### Environment

- Simulator and dependency versions.
- Robot embodiment.
- Environment initialization.
- Task registration.
- Task reset mechanism.
- Initial state sampling.
- Episode termination.
- Task success conditions.
- Evaluation tracks.
- Maximum episode length.
- Control frequency.
- Environment stepping mechanism.

### Observation Space

Determine the exact observation schema, including:

- Primary RGB camera.
- Secondary RGB camera.
- Wrist RGB camera.
- Robot proprioception.
- End-effector state.
- Gripper state.
- Natural-language instruction.
- Image resolution.
- Image preprocessing.
- Coordinate conventions.

### Action Space

Verify:

- Action dimension.
- End-effector translation representation.
- Rotation representation.
- Gripper representation.
- Absolute versus relative versus delta actions.
- Action normalization statistics.
- Action clipping.
- Coordinate transformations.
- Action execution frequency.

Do not assume VLABench uses the same action representation as LIBERO or RoboCasa.

### Checkpoint

Inspect the official checkpoint:

```text
VLABench/pi05-primitive-10task
```

Determine:

- Checkpoint format.
- Model architecture.
- Required inference framework.
- Model configuration.
- Prediction horizon.
- Default execution horizon.
- Denoising steps.
- Normalization assets.
- Input and output transforms.
- Supported precision.
- JAX or PyTorch compatibility.

Always use the normalization assets associated with the target checkpoint.

If checkpoint conversion is necessary, investigate and validate numerical equivalence before using the converted model for AAC or DUAC.

---

## 6. Phase 3 — Target Evaluation Tasks

Support all ten Primitive Tasks associated with the official checkpoint.

| Task ID | Research characteristics |
|---|---|
| add_condiment | Rotation, pouring, manipulation |
| insert_flower | Precision insertion and alignment |
| select_book | Pulling and object extraction |
| select_chemistry_tube | Fine-grained manipulation |
| select_drink | Object grounding and retrieval |
| select_fruit | Basic object manipulation |
| select_mahjong | Texture and symbol recognition |
| select_painting | Visual grounding and target interaction |
| select_poker | Fine-grained grasping |
| select_toy | Object geometry and grasp variation |

Verify all task IDs against the official task registry.

Implement support for evaluating:

- A single task.
- A selected subset of tasks.
- All ten tasks.
- A single evaluation track.
- Multiple evaluation tracks.

The initial evaluation should prioritize Track 1.

Subsequent evaluations should support:

| Track | Description |
|---|---|
| T1 | In-distribution |
| T2 | Cross-category generalization |
| T3 | Common sense reasoning |
| T4 | Semantic instruction understanding |
| T6 | Unseen texture |

Verify the exact track identifiers from the installed benchmark version.

Do not confuse the current repository's T6 Unseen Texture with the original paper's T6 Long-Horizon Task Learning.

---

## 7. Phase 4 — Benchmark Adapter Implementation

Implement a VLABench adapter that follows the existing benchmark abstraction whenever possible.

The desired architecture is:

```text
Existing Research Framework
|
|-- Shared VLA Policy
|-- Shared AAC Algorithm
|-- Shared DUAC Algorithm
|-- Shared Horizon Selection
|-- Shared Metrics and Logging
|
|-- Benchmark Adapters
    |-- LIBERO
    |-- RoboCasa
    |-- VLABench (NEW)
```

The VLABench adapter should handle:

1. Environment initialization.
2. Task loading.
3. Episode initialization.
4. Observation extraction.
5. Observation preprocessing.
6. Policy input conversion.
7. Action denormalization.
8. Action representation conversion.
9. Environment stepping.
10. Success detection.
11. Episode termination.
12. Video recording.
13. Evaluation metric collection.

Maintain separation between benchmark-specific logic and algorithm-specific logic.

Do not duplicate AAC or DUAC implementations inside the VLABench adapter.

Preserve the existing LIBERO and RoboCasa functionality.

---

## 8. Phase 5 — Integrate Existing AAC and DUAC

### 8.1. Vanilla Baseline

Load the official pretrained π₀.₅ checkpoint.

Use the official inference configuration and default execution horizon.

Reproduce the official evaluation behavior as closely as possible before enabling adaptive chunking.

### 8.2. Fixed Action Chunking

Support configurable execution horizons, subject to the checkpoint's actual prediction horizon.

Suggested candidates:

```yaml
fixed_horizons:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
```

Exclude values greater than the actual prediction horizon.

Keep the prediction horizon unchanged.

### 8.3. AAC

Reuse the existing AAC implementation.

Preserve:

- Action sampling mechanism.
- Number of stochastic samples.
- Entropy estimation.
- Uncertainty aggregation.
- Adaptive horizon selection.
- Thresholding mechanism.
- Existing hyperparameters.
- Action selection strategy.

Ensure that AAC receives the correct VLABench observation and action representations.

Preserve the original mathematical formulation.

### 8.4. DUAC

Reuse the existing DUAC implementation.

Preserve:

- Flow-field uncertainty estimation.
- Jacobian or derivative computation as implemented in the existing method.
- Uncertainty aggregation.
- Flow Matching integration.
- Adaptive horizon selection.
- Existing hyperparameters.

Investigate whether checkpoint loading, compilation, gradient disabling, mixed precision, or inference mode interferes with DUAC's required derivative computations.

Do not silently substitute covariance, variance, or another uncertainty estimator for the existing DUAC formulation.

Do not modify the mathematical formulations of AAC or DUAC unless a compatibility issue makes this unavoidable. Document and justify any necessary change.

---

## 9. Phase 6 — Unified Closed-Loop Inference

For every method, use the same frozen VLA checkpoint and environment configuration.

The conceptual evaluation loop should follow:

```python
obs = env.reset()

while not done:
    policy_input = preprocess(obs)

    prediction = policy.infer(policy_input)

    horizon = method.select_horizon(
        prediction=prediction,
        observation=policy_input,
        policy=policy,
    )

    horizon = validate_horizon(horizon)

    for action in prediction.actions[:horizon]:
        env_action = postprocess(action)
        obs, reward, done, info = env.step(env_action)

        if done:
            break
```

This pseudocode is conceptual.

Adapt it to the existing project interfaces. AAC and DUAC may require access to stochastic samples, noise, flow fields, or intermediate model states rather than only the final predicted action sequence.

The implementation must preserve their existing inference mechanisms.

Ensure:

- No stale observations between inference calls.
- No execution beyond episode termination.
- No off-by-one errors.
- No invalid execution horizons.
- No unintended model state changes.
- No incorrect action buffer reuse.
- No double normalization or denormalization.
- No use of privileged simulator state for uncertainty estimation unless explicitly identified as an oracle baseline.

---

## 10. Phase 7 — Evaluation Metrics

Collect metrics at action-chunk, episode, task, track, and benchmark levels.

### Task Performance

- Success Rate.
- Progress Score, where officially supported.
- Intention Score, where officially supported.
- Episode length.
- Number of completed episodes.
- Failure and timeout counts.

### Adaptive Action Chunking

- Prediction horizon.
- Execution horizon.
- Mean execution horizon.
- Standard deviation of execution horizon.
- Minimum and maximum execution horizon.
- Execution horizon distribution.
- Number of policy inference calls.
- Number of executed actions.
- Number of stochastic action samples.
- Number of action expert evaluations.

### Uncertainty Quantification

- Per-chunk uncertainty.
- Per-action uncertainty, if available.
- Uncertainty before horizon selection.
- Uncertainty distribution.
- Entropy statistics for AAC.
- Flow-field uncertainty statistics for DUAC.
- Correlation between uncertainty and subsequent failure.

If failure prediction metrics are implemented, define the prediction target and future time window explicitly.

### Computational Efficiency

- Observation preprocessing time.
- VLA backbone inference time.
- Action expert inference time.
- Uncertainty estimation time.
- Horizon selection time.
- Action postprocessing time.
- Environment stepping time.
- Total policy computation time.
- End-to-end episode wall-clock time.
- Peak GPU memory usage.
- GPU and CPU device configuration.

Use synchronized GPU timing when necessary.

Separate model warm-up and compilation overhead from steady-state inference latency.

Do not use inference-call reduction alone as evidence of computational acceleration.

Account for AAC's stochastic sampling overhead and DUAC's derivative computation overhead.

---

## 11. Phase 8 — Experiment Logging

Reuse the project's existing logging and output conventions.

If necessary, extend them to save:

### Episode-level metrics

```text
benchmark
checkpoint
method
task
track
episode_id
seed
success
progress_score
intention_score
episode_steps
inference_calls
average_execution_horizon
total_policy_time_ms
uncertainty_time_ms
episode_wall_time_ms
peak_gpu_memory_mb
```

### Chunk-level metrics

```text
episode_id
chunk_id
simulation_step
prediction_horizon
selected_execution_horizon
executed_horizon
uncertainty
inference_time_ms
uncertainty_time_ms
horizon_selection_time_ms
sample_count
```

When practical, support saving:

- Predicted action trajectories.
- Executed action trajectories.
- Per-action uncertainty.
- Robot states.
- Episode videos.
- Failure examples.

Large trajectory and video outputs should be configurable rather than mandatory.

Support resumable evaluation without duplicating completed episodes.

Save the resolved experiment configuration and software versions alongside the results.

---

## 12. Phase 9 — Experimental Protocol

Ensure fair comparisons across all methods.

Control the following variables:

- Identical pretrained checkpoint.
- Identical task definitions.
- Identical initial environment states.
- Identical evaluation seeds.
- Identical episode termination conditions.
- Identical maximum episode lengths.
- Identical action representations.
- Identical prediction horizons.
- Identical model precision.
- Identical base denoising configuration unless required by the method.
- Identical hardware and timing methodology.

Use matched episode initializations across methods.

Control policy random number generation explicitly and document how random sampling differs between Vanilla, AAC, and DUAC.

Reuse existing AAC and DUAC hyperparameters as the initial configuration.

Do not tune hyperparameters on the final evaluation episodes.

If hyperparameter calibration is required, use separate calibration episodes.

### Stage A: Smoke Test

Tasks:

```yaml
tasks:
  - select_fruit

methods:
  - vanilla
  - aac
  - duac

episodes_per_task: 2
```

### Stage B: Pilot Experiment

Tasks:

```yaml
tasks:
  - select_fruit
  - add_condiment
  - insert_flower
  - select_book
  - select_chemistry_tube

methods:
  - vanilla
  - aac
  - duac

episodes_per_task: 10
```

### Stage C: Full In-distribution Evaluation

```yaml
benchmark: vlabench
checkpoint: VLABench/pi05-primitive-10task

tasks: primitive_10
track: T1

methods:
  - vanilla
  - fixed_chunking
  - aac
  - duac

episodes_per_task: 50
```

### Stage D: Robustness Evaluation

```yaml
benchmark: vlabench
checkpoint: VLABench/pi05-primitive-10task

tasks: primitive_10

tracks:
  - T2
  - T3
  - T4
  - T6

methods:
  - vanilla
  - aac
  - duac

episodes_per_task: 50
```

These YAML examples describe desired experimental configurations, not necessarily existing configuration schemas.

Adapt them to the project without unnecessarily replacing the existing configuration system.

---

## 13. Phase 10 — Visualization and Result Analysis

Implement or extend visualization scripts to generate:

1. Success Rate comparison across methods.
2. Per-task Success Rate.
3. Per-track Success Rate.
4. Execution horizon distribution.
5. Uncertainty distribution.
6. Inference latency comparison.
7. Computational overhead breakdown.
8. Success Rate versus computational cost.
9. Uncertainty versus execution horizon.
10. Uncertainty and execution horizon over episode time.
11. Successful versus failed episode uncertainty.
12. Cross-benchmark comparison across LIBERO, RoboCasa, and VLABench.

Include confidence intervals where appropriate.

For cross-benchmark comparisons, account for differences in episode length, action frequency, and evaluation protocols.

Export publication-quality plots in PDF and PNG formats.

Export numerical results in CSV or the project's existing structured format.

Do not fabricate experimental results.

---

## 14. Phase 11 — Validation and Testing

Before running large-scale experiments, validate:

- Checkpoint loading.
- Observation compatibility.
- Action dimension and representation.
- Normalization and denormalization.
- Environment reset.
- Environment stepping.
- Episode termination.
- Vanilla inference.
- Fixed horizon execution.
- AAC stochastic sampling.
- AAC uncertainty estimation.
- DUAC flow-field uncertainty.
- DUAC derivative computation.
- Adaptive horizon boundaries.
- Metric correctness.
- GPU memory usage.
- Reproducibility.
- Resume functionality.

Verify that AAC and DUAC preserve the mathematical behavior of the existing implementations.

Run regression tests to ensure LIBERO and RoboCasa functionality has not been broken.

Compare the reproduced Vanilla baseline against official checkpoint results under the corresponding evaluation protocol.

If results differ substantially, investigate the discrepancy before launching full-scale experiments.

---

## 15. Engineering Constraints

- No training.
- No fine-tuning.
- No modification of pretrained weights.
- No replacement of the existing AAC algorithm.
- No replacement of the existing DUAC algorithm.
- No unnecessary duplication.
- No breaking changes to LIBERO or RoboCasa.
- No hard-coded machine-specific paths.
- No fabricated metrics.
- No hidden modifications to benchmark success criteria.
- No evaluation using privileged information as policy input.
- Maintain modularity, reproducibility, and documentation.

Use isolated dependencies or separate benchmark environments if VLABench introduces dependency conflicts.

Do not download the full reference training dataset unless it is demonstrably required for inference or evaluation.

Avoid expensive full-scale experiments until smoke tests and baseline validation are completed.

---

## 16. Expected Deliverables

After completing the implementation, provide:

1. Existing codebase architecture analysis.
2. VLABench integration design.
3. Working VLABench environment adapter.
4. Official π₀.₅ checkpoint integration.
5. Vanilla evaluation support.
6. Fixed Action Chunking support.
7. AAC evaluation support.
8. DUAC evaluation support.
9. Experimental configurations.
10. Evaluation and visualization scripts.
11. Validation and regression tests.
12. Installation instructions.
13. Checkpoint download instructions.
14. Exact runnable evaluation commands.
15. An explanation of modified and newly created files.
16. Smoke-test results.
17. Known limitations and unresolved issues.

Create documentation using the project's existing documentation conventions.

If no appropriate documentation location exists, create:

```text
docs/VLABENCH_INTEGRATION.md
docs/VLABENCH_EXPERIMENT_PROTOCOL.md
```

---

## 17. Final Execution Instructions

Start by inspecting the current repository and understanding