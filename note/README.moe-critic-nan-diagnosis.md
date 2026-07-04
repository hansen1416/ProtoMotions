# Diagnosis: Critic-Only NaN Crash in MoE Actor Training (`mlp_moe.py`)

**Setup**: 6×GPU server (A40, 46GB each)
**Command**: `python -u protomotions/train_agent.py --robot-name smpl_mor --simulator isaacgym --experiment-path examples/experiments/mimic/mlp_moe.py --experiment-name hhi_moe_20946_2shape --motion-file /workspace/hhi_stage1_merged6/hhi_stage1_41892_slurmrank.pt --num-envs 6144 --batch-size 24576 --ngpu 6 --use-wandb ...`

**Symptom**:
```
Epoch 0, collecting data...
[rank0]: AssertionError: NaN or Inf in value
  at protomotions/agents/base_agent/agent.py:380, in collect_rollout_step
```
Crashed deterministically within seconds of starting rollout collection, on a from-scratch run (no `--checkpoint`). `action`/`mean_action`/`neglogp` (actor outputs) always passed the finiteness check first — only the critic's `value` output was ever NaN.

---

## Initial (wrong) hypothheses ruled out by static reading

- **Data corruption**: the merged Stage-1 dataset was pre-verified NaN/Inf-free across all tensor keys. Confirmed clean again live (see below).
- **Zero-variance `morphology_obs` dims**: the two body shapes share one beta vector (only gender differs), so 10 of 11 morphology dims have ~zero population variance. `RunningMeanStd.normalize()` has `epsilon=1e-5` inside the `sqrt`, so this can't literally divide by zero — confirmed not the cause.
- **Uninitialized-DDP-broadcast of LazyLinear params**: ruled out — `agent.py`'s `setup()` runs a dummy forward pass to materialize all `LazyLinear`/`RunningMeanStd` modules *before* `fabric.setup()` wraps the model in DDP.
- **MoE-specific bug** (gate/expert blending, load-balancing loss): the critic is an unmodified `MLPWithConcat` (same 4×1024 config as the working `mlp.py` baseline) — the MoE actor and critic share no parameters, and the crash is 100% in the critic's own forward pass.

---

## Diagnostic Instrumentation & Key Findings

Live GPU access let us reproduce the crash on demand at full scale, so we added temporary print instrumentation at three checkpoints in `protomotions/agents/base_agent/agent.py` and re-ran:

**1. At crash time** (`collect_rollout_step`, right before the `isfinite` assert):
```
obs[max_coords_obs]      all_finite=True  min=-0.94   max=6142.99
obs[mimic_target_poses]  all_finite=True  min=-6142.99 max=228.07
obs[previous_actions]    all_finite=True  min=0.0     max=0.0
obs[morphology_obs]      all_finite=True  min=-1.0    max=1.0
critic RunningMeanStd    all_finite=True  (count=73729, var up to ~3.15e6)
critic.mlp[0] weight_finite=True bias_finite=True
critic.mlp[2] weight_finite=True bias_finite=False   ← corrupted
critic.mlp[4] weight_finite=True bias_finite=True
critic.mlp[6] weight_finite=True bias_finite=True
critic.mlp[8] weight_finite=True bias_finite=True
value: all 6144 rows NaN
```
- All raw observations and normalization stats were finite. Only `critic.mlp[2]`'s **bias** (the 2nd hidden `Linear(1024→1024)`) was NaN — weights were fine, every other layer was fine.
- The **entire batch** of 6144 `value` outputs was NaN uniformly. A per-environment physics divergence would only corrupt the rows for the affected envs; a uniformly-NaN batch means a *parameter* is corrupted, not the input.
- Identical corruption (same layer, same pattern) appeared on every rank that crashed — but not on rank 0.

**2. Right after the warmup materialization pass, before `create_optimizers()`/DDP wraps anything**: `critic.mlp[2]` bias was **finite on all 6 ranks**, including rank 0.

**3. Right after `create_optimizers()` returns (DDP setup complete, before any further forward pass)**: still **finite on all 6 ranks** (confirmed the DDP-wrapped `self.critic` and the raw `model._critic` share the exact same bias tensor object — no divergence introduced by DDP's constructor-time parameter broadcast).

**4. Right after the second, real forward pass in `fit()`** (`self.model(obs_td)` on the real post-`env.reset()` observations, used to auto-register experience-buffer keys — still in `train()` mode, so `RunningMeanStd.record_moments()` fires):
```
rank=0  critic.mlp[2] bias_finite=True   value_finite=True    ← clean (broadcast source)
rank=1  critic.mlp[2] bias_finite=False  value_finite=False   ← corrupted (receiver)
rank=2  critic.mlp[2] bias_finite=False  value_finite=False   ← corrupted (receiver)
rank=3  critic.mlp[2] bias_finite=False  value_finite=False   ← corrupted (receiver)
rank=4  critic.mlp[2] bias_finite=False  value_finite=False   ← corrupted (receiver)
rank=5  critic.mlp[2] bias_finite=False  value_finite=False   ← corrupted (receiver)
```
This pinned the corruption to *this exact call* — specifically to `RunningMeanStd.record_moments()`, the only code that performs collective communication during a plain forward pass. The pattern (source rank clean, every receiving rank identically corrupted) is the signature of a bad broadcast, not a numerics/logic bug.

Small-scale repros (256 envs, 1–2 GPUs, world_size ≤ 2) never hit this even after 14+ epochs — consistent with a collective-communication bug that only manifests with enough ranks/traffic.

---

## Root Cause

`RunningMeanStd.record_moments()` (`protomotions/agents/utils/normalization.py`) synchronizes its `mean`/`var`/`count` buffers across ranks using:
```python
updated_mean = self.fabric.broadcast(self.mean, src=0)
```
Lightning Fabric's `DDPStrategy.broadcast()` implements this as:
```python
def broadcast(self, obj, src=0):
    obj = [obj]
    torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
    return obj[0]
```
`broadcast_object_list` is a **pickle-based, generic-Python-object broadcast** — designed for small, arbitrary (possibly non-tensor) objects — not a native NCCL tensor broadcast. It round-trips large CUDA tensors through serialize/deserialize on every call. This function is invoked **every rollout step, once per `RunningMeanStd` instance** (the actor's and the critic's normalizers each call it independently), so under real training load (6 ranks, large tensors, heavy concurrent GPU work from IsaacGym's own physics simulation) it reliably corrupted GPU memory on the receiving ranks — landing this time on the critic's `mlp[2]` bias, which happened to sit nearby in the allocator.

This is **not MoE-specific** — `RunningMeanStd` is shared code used by every experiment config (including the flat-concat `mlp.py` baseline). The MoE run simply was the first workload to exercise this exact rank count/env count/traffic combination.

---

## Fix

**File**: `protomotions/agents/utils/normalization.py`

```diff
-        # Broadcast updated parameters to all ranks
-        updated_mean = self.fabric.broadcast(self.mean, src=0)
-        updated_var = self.fabric.broadcast(self.var, src=0)
-        updated_count = self.fabric.broadcast(self.count, src=0)
-
-        self.mean.copy_(updated_mean)
-        self.var.copy_(updated_var)
-        self.count.fill_(updated_count.item())
+        # Broadcast updated parameters to all ranks.
+        # NOTE: fabric.broadcast() round-trips through torch.distributed.broadcast_object_list
+        # (pickle-based generic object broadcast), which is not a reliable way to repeatedly
+        # broadcast large CUDA tensors under NCCL. Use a native in-place tensor broadcast instead.
+        if torch.distributed.is_available() and torch.distributed.is_initialized():
+            torch.distributed.broadcast(self.mean, src=0)
+            torch.distributed.broadcast(self.var, src=0)
+            torch.distributed.broadcast(self.count, src=0)
```

Replaces the pickle-based object broadcast with a direct, in-place `torch.distributed.broadcast()` on each buffer — the standard, well-tested way to synchronize tensors across ranks under NCCL.

---

## Validation

Re-ran the exact crashing command (6144 envs, 6 GPU, real data) after the fix:
- Epoch 0 completed fully (data collection **and** the optimization phase) — previously it never got past the first few rollout steps.
- Epoch 1 collection was underway with no NaN on any rank.
- All temporary debug instrumentation was removed from `agent.py` afterward (`git diff` on that file is empty); only `normalization.py` carries the real fix.
