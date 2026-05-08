# RCAR-VLA Implementation Plan

## Objective

Turn the current `KANFlow-VLA` codebase from a pure action predictor into a reliability-aware VLA that can:

- act on ordinary manipulation instructions
- ask for clarification when the instruction is ambiguous
- abstain when the request is impossible or unsafe
- recover after perturbations or execution failures
- obey a revised instruction mid-episode without resetting

This plan is scoped to the current repo and should be implemented on top of the existing MetaWorld pipeline first. The first public benchmark can be `MetaWorld-RCAR`. If that works, the same interfaces can later be ported to LIBERO or a richer tabletop benchmark.

## Why This Fits This Repo

The current code already has the right seams:

- `kanflow_vla/data/metaworld_dataset.py` owns sample construction and language conditioning
- `kanflow_vla/model/kanflow_vla.py` builds a single fused condition vector that is a natural place to attach behavior heads
- `kanflow_vla/losses.py` already combines multiple training terms
- `kanflow_vla/train.py` already has periodic validation/checkpoint logic
- `kanflow_vla/eval_metaworld.py` already owns rollout loops and can be extended into event-driven evaluation

We do not need to rewrite the action core. We should keep the existing flow-matching policy and add reliability behavior around it.

## Design Decisions

### 1. Keep the current policy as the action core

The current `ConsistencyFlowMatching` action head stays in place. Recovery should initially reuse the same action decoder, trained on perturbed trajectories, instead of introducing a second action model.

### 2. Add explicit behavior heads

Attach small heads to the fused condition vector:

- `mode_head`: predicts `act`, `ask`, `abstain`, or `recover`
- `failure_head`: predicts failure or ambiguity type
- `progress_head`: predicts normalized task progress in `[0, 1]`

Suggested labels:

- `mode`: `act=0`, `ask=1`, `abstain=2`, `recover=3`
- `failure`: `none=0`, `ambiguity=1`, `wrong_object=2`, `grasp_miss=3`, `occlusion=4`, `unreachable=5`, `unsafe=6`, `contradiction=7`, `unknown=8`

### 3. Use templated utterances for v1

Do not add a full language decoder first. That will slow the project down and create a second research problem.

For the first version:

- `ask` emits a templated clarification message from the predicted ambiguity type
- `abstain` emits a templated refusal message from the predicted failure type

Examples:

- `"Which object do you mean?"`
- `"I cannot complete that safely."`
- `"The target is ambiguous. Please clarify the object."`

The research novelty is the behavior, not open-ended text generation.

### 4. Treat "revise" as instruction-state update, not a fifth mode

`revise` should be handled by updating the current instruction tokens during rollout. The model does not need a separate `revise` action head. The benchmark should test whether success recovers after the instruction changes.

### 5. Replace dataset duplication with sampler-based balancing

The current balancing logic duplicates short-task windows until every task reaches the largest task count. That is simple but expensive and likely to overfit short tasks.

Before the first long RCAR run, move balancing to a sampler:

- keep the dataset as unique windows
- assign each sample a per-task weight
- use `WeightedRandomSampler` or a custom balanced batch sampler

That preserves task balance without inflating epoch length 5-6x.

## New Benchmark: MetaWorld-RCAR

Each base task should support six episode types:

- `normal`: ordinary instruction following
- `counterfactual`: same scene, different instruction target
- `ambiguous`: multiple objects satisfy the request, model should ask
- `correction`: instruction changes mid-trajectory
- `perturbation`: object or state is disturbed, model should recover
- `impossible_or_unsafe`: instruction is contradictory, impossible, or unsafe, model should abstain

Primary metrics:

- `task_success`
- `language_fidelity`
- `clarification_success`
- `recovery_success`
- `unsafe_action_rate`
- `correction_success`
- `correction_latency_steps`
- `false_ask_rate`
- `false_abstain_rate`

Recommended headline metric:

`rcar_score = 0.30 * task_success + 0.20 * language_fidelity + 0.20 * recovery_success + 0.15 * clarification_success + 0.15 * (1 - unsafe_action_rate)`

## File-By-File Plan

### Modify `kanflow_vla/model/kanflow_vla.py`

Extend `KANFlowVLAOutput` with:

- `mode_logits`
- `failure_logits`
- `progress`

Add a small reliability head block after `_encode_condition()`:

- `nn.Linear(d_model, d_model)` + `GELU`
- separate output heads for mode, failure, progress

Suggested API changes:

- `forward(...)` should accept optional labels for `mode`, `failure`, and `progress`
- `predict_action(...)` should gain a companion method like `predict_behavior(...)`
- add `predict_step(...)` that returns both action and behavior outputs for rollout code

Important detail:

- keep `predict_action()` available for compatibility
- new eval code should call `predict_step()` instead

### Add `kanflow_vla/model/reliability_heads.py`

Create a small module to keep behavior heads out of the main model file.

Contents:

- `ReliabilityHeads`
- label constants or enums for `mode` and `failure`
- helper to convert logits to structured decisions

Keep this small and explicit. This module should be easy to ablate independently from the flow model.

### Modify `kanflow_vla/losses.py`

Extend `KANFlowLoss` to include:

- `mode_loss`: cross entropy
- `failure_loss`: cross entropy
- `progress_loss`: smooth L1 or MSE

Configurable weights:

- `lambda_mode`
- `lambda_failure`
- `lambda_progress`

Recommended starting weights:

- `lambda_mode = 1.0`
- `lambda_failure = 0.5`
- `lambda_progress = 0.25`

The output dict should log all auxiliary losses separately.

### Modify `kanflow_vla/data/metaworld_dataset.py`

Add RCAR supervision to each sample.

New fields returned by `__getitem__()`:

- `instruction_text`
- `instruction_variant`
- `mode_label`
- `failure_label`
- `progress_label`
- `event_mask`

New behaviors to implement in the dataset:

- generate instruction variants for each clip
- generate ambiguity labels when the language is intentionally underspecified
- generate abstain labels for impossible or contradictory instruction variants
- compute normalized progress labels from timestep within episode

Important implementation choice:

- keep `normal` and `counterfactual` variants entirely offline
- keep `ambiguous` and `impossible_or_unsafe` variants as synthetic language rewrites first
- do not try to simulate recovery in the offline dataset alone

Also change task balancing:

- remove full duplication in `_balance_samples()`
- return unique samples only
- expose per-sample weights or per-task counts for sampler construction

### Add `kanflow_vla/data/rcar_language.py`

Move all language variant generation here so `metaworld_dataset.py` stays readable.

Functions to add:

- `build_instruction_text(task_name, variant, metadata=None)`
- `sample_instruction_variant(task_name, rng)`
- `build_clarification_target(...)`
- `build_impossible_instruction(...)`

Variant set for v1:

- `normal`
- `counterfactual`
- `ambiguous`
- `impossible`
- `unsafe`
- `corrected`

This file should own the text templates and label mapping.

### Add `kanflow_vla/data/balanced_sampler.py`

Implement one of:

- `WeightedRandomSampler` helper
- custom balanced sampler that targets equal task frequency per epoch

The sampler should let you control epoch size directly without multiplying dataset length.

### Modify `kanflow_vla/train.py`

Training loop changes:

- pass the new labels into the model or loss function
- log `mode_acc`, `failure_acc`, and `progress_mae`
- keep existing MT10 validation
- add periodic RCAR validation episodes

New CLI/config parameters:

- `--rcar-val-every`
- `--rcar-val-rollouts`
- `--sampler balanced|uniform`
- `--epoch-size`

Checkpoint policy:

- `best_eval.pt` remains the default deployment checkpoint
- add `best_rcar.pt` for the best RCAR score

Also add a staged training option:

- `phase=policy_only`
- `phase=policy_plus_behavior`
- `phase=recovery_finetune`

### Add `kanflow_vla/eval_rcar.py`

This is the key new evaluation entry point.

Responsibilities:

- create RCAR episode variants
- inject instruction corrections mid-rollout
- inject perturbations at controlled steps
- track `ask`, `abstain`, `recover`, and correction outcomes
- compute RCAR metrics and save a structured JSON summary

Core functions:

- `run_rcar_rollout(...)`
- `inject_instruction_revision(...)`
- `inject_perturbation(...)`
- `score_language_fidelity(...)`
- `aggregate_rcar_results(...)`

This script should be separate from `eval_metaworld.py`. The existing script can remain the plain success-rate baseline.

### Modify `kanflow_vla/eval_metaworld.py`

Keep this as the plain baseline evaluator, but refactor shared pieces into helpers so `eval_rcar.py` can reuse:

- environment construction
- observation preprocessing
- temporal buffers
- task tokenization

Do not turn `eval_metaworld.py` into a giant mixed benchmark file.

### Add `kanflow_vla/metrics.py`

Add reusable metric helpers:

- classification accuracy
- calibration or entropy logging for `mode`
- language fidelity score
- correction latency
- unsafe action rate

This keeps `train.py` and `eval_rcar.py` cleaner.

### Add `kanflow_vla/configs/rcar_metaworld.yaml`

Create a dedicated config instead of overloading the current training config.

Suggested sections:

- `data.rcar_variants`
- `data.sampler`
- `model.reliability`
- `loss.lambda_mode`
- `loss.lambda_failure`
- `loss.lambda_progress`
- `eval.rcar`

Keep `metaworld.yaml` as the baseline policy config.

### Add `scripts/build_rcar_manifest.py`

Build a manifest of RCAR episode specifications.

Output:

- a JSONL or JSON manifest listing task, demo, timestep, instruction variant, perturbation schedule, correction step, and expected labels

This makes the benchmark reproducible and easier to inspect than burying all augmentation logic inside the dataloader.

### Add `scripts/run_rcar_canary.sh`

Small command wrapper for the first real experiment:

- short run
- balanced sampler
- RCAR validation every epoch
- small number of rollouts

This should be the command you trust before another long GPU job.

## Phase Plan

### Phase 0: Clean Refactor

Goal:

- separate policy code from reliability code
- remove dataset-length inflation
- prepare eval reuse

Deliverables:

- sampler-based balancing
- `reliability_heads.py`
- `metrics.py`
- refactored eval helpers

Exit criteria:

- baseline training still works
- no checkpoint/load regression
- canary train + baseline eval pass

### Phase 1: Behavior Heads on Offline Data

Goal:

- learn `act`, `ask`, `abstain`, and progress prediction from synthetic instruction variants and offline clips

Deliverables:

- auxiliary heads wired into model and loss
- dataset emits variant labels
- RCAR evaluator supports `normal`, `ambiguous`, and `impossible`

Exit criteria:

- `mode_acc > 90%` on held-out synthetic variants
- `false_ask_rate < 10%` on normal episodes
- `false_abstain_rate < 5%` on normal episodes

### Phase 2: Mid-Episode Correction

Goal:

- verify the model can obey instruction updates without environment reset

Implementation:

- inject correction text mid-rollout
- retokenize the current instruction and continue
- score whether the final target matches the revised instruction

Exit criteria:

- correction success materially exceeds baseline policy
- correction latency is stable and bounded

### Phase 3: Recovery Training

Goal:

- learn to replan after disturbances

Data strategy:

- scripted perturbations in simulation
- short recovery trajectories collected with scripted controller or DAgger
- label `mode=recover` near the perturbation window

Exit criteria:

- recovery success clearly above non-recovery baseline
- no major collapse in ordinary task success

### Phase 4: RCAR Benchmark Release

Goal:

- produce the benchmark result that supports the paper claim

Deliverables:

- `eval_rcar.py`
- benchmark manifest
- aggregated JSON result format
- ablations:
  - no behavior heads
  - no correction training
  - no recovery data
  - no abstain head

## First Experiment Order

Do not jump straight to full RCAR training. Run these in order:

1. Restore sampler-based balancing and confirm baseline MT10 still trains.
2. Add behavior heads and train only on `normal`, `ambiguous`, and `impossible` variants.
3. Add correction episodes in eval before training them, so you can measure the baseline gap.
4. Add correction training.
5. Add perturbation injection and recovery data collection.
6. Only then launch a longer full RCAR run.

## Canary Run Recommendation

Use a short canary before any multi-day job:

- `epochs=3`
- `eval_every=1`
- `rcar_val_every=1`
- `val_rollouts=1`
- `rcar_val_rollouts=1`
- balanced sampler with fixed `epoch_size`

Must verify:

- `best_eval.pt` still updates
- `best_rcar.pt` saves separately
- `mode` predictions are not degenerate
- `ask` and `abstain` do not dominate ordinary episodes

## Go / No-Go Criteria For Long Training

Proceed to a long run only if all are true:

- baseline MT10 success does not materially regress
- behavior heads train stably and reach non-trivial validation accuracy
- RCAR canary metrics improve over the baseline policy
- epoch time is acceptable after sampler change
- no checkpoint incompatibility remains between train and eval

Do not proceed if:

- `ask` becomes a crutch on normal episodes
- `abstain` spikes on recoverable tasks
- task balance still relies on dataset duplication
- RCAR gains come from templated shortcuts without real control improvement

## Minimal First Claim

The first claim should be modest and defensible:

"`RCAR-VLA` extends the current KANFlow-VLA policy with explicit act/ask/abstain/recover control heads and a benchmark for correction, abstention, and recovery under mid-episode language changes and physical perturbations."

Only make broader public claims after the benchmark is real and the ablations support them.
