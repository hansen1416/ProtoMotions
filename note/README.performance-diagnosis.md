# Performance Diagnosis: FiLM Training Slowdown

**Setup**: 4×GPU server, 96 CPU cores, 503GB RAM
**Command**: `python -u protomotions/train_agent.py --robot-name smpl_mor --simulator isaacgym --experiment-path examples/experiments/mimic/mlp_film.py --experiment-name hhi_film_1024_motion --motion-file /workspace/merged4/humos_slurmrank.pt --num-envs 8192 --batch-size 32768 --ngpu 4`

---

## Diagnostic Commands & Key Findings

**1. Training throughput**
```
trainer/global_step ≈ 4500 in 39h → ~31 sec/epoch
times/fps_total = 32,500 env-steps/sec (total across 4 ranks = 8,125/rank)
Expected: >100,000 fps total → ~20× slower than expected
```

**2. GPU utilization**
```bash
nvidia-smi dmon -s u -d 2 -c 30
```
```
GPU 0: sm=97-100%   ← saturated
GPU 1: sm=46-57%    ← moderate
GPU 2: sm=0%        ← completely idle
GPU 3: sm=0%        ← nearly idle
```

**3. CPU utilization**
```bash
top -b -n 3 -d 2 | head -60
```
```
4 × pt_main_thread at ~100% each
Total CPU: ~13% of 96 cores → CPU is NOT the bottleneck
```

**4. I/O check**
```bash
iostat -x 1 5
```
```
iowait ≈ 0% → disk is NOT the bottleneck
```

**5. Confirm GPU↔process mapping**
```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
nvidia-smi --query-gpu=index,uuid --format=csv
```
```
Rank 0 (PID 3768) → GPU 0, 26GB sim  ✓
Rank 1 (PID 3965) → GPU 1, 26GB sim  ✓
Rank 2 (PID 3966) → GPU 2, 26GB sim  ✓
Rank 3 (PID 3967) → GPU 3, 26GB sim  ✓
All repeated GPU-0/380MB entries = normal NCCL peer-access buffers
```
GPU assignment is correct — yet GPUs 2 and 3 are still idle → bottleneck is inside the simulator.

**6. Root cause in code**
`protomotions/simulator/isaacgym/simulator.py:99-100`:
```python
if self.headless is True:
    self._graphics_device_id = 0   # ← forces ALL ranks to GPU 0
```
`gym.create_sim(compute_device_id, graphics_device_id)` — IsaacGym uses the graphics device for internal state tensor management even in headless mode. With all 4 ranks pointing their graphics context at GPU 0, that GPU handled render-path operations for all simulations. GPUs 2–3 ran their physics briefly then stalled waiting on GPU 0.

---

## Key Points

1. **GPU assignment was correct** — sims were on GPUs 0–3 as expected; this red herring was ruled out by running `--query-compute-apps` with `gpu_uuid`.
2. **CPU/IO were not bottlenecks** — only 4 cores active, zero iowait.
3. **The bug**: `_graphics_device_id = 0` for all headless ranks funneled IsaacGym's internal state management through GPU 0, serializing all 4 ranks on that one GPU.
4. **Expected fix**: all 4 GPUs should show ~equal SM utilization and fps_total should increase toward 100k+.

---

## Code Change

**File**: `protomotions/simulator/isaacgym/simulator.py`

```diff
-        self._graphics_device_id = device_index
-        if self.headless is True:
-            self._graphics_device_id = 0
+        # In multi-GPU DDP, each rank must use its own GPU as the graphics device.
+        # Forcing graphics_device_id=0 for all headless ranks routes IsaacGym's internal
+        # state management through GPU 0, bottlenecking all other ranks on that one GPU.
+        self._graphics_device_id = device_index
```
