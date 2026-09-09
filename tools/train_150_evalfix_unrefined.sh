#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
# Complete the architecture/data comparison with three unrefined-HUMOS runs.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash tools/train_150_evalfix_unrefined.sh [--dry-run] [--all-seeds]

Three independent trainings, one per GPU (default GPUs: 0,1,2):
  F: temporal MLP on unrefined HUMOS          (counterpart to A)
  G: basic attention on unrefined HUMOS       (counterpart to B)
  H: slot/type + actor-only AdaLN on unrefined HUMOS (counterpart to D)
Slot/type-only on unrefined HUMOS already exists as E; it is not repeated.

Defaults: seed 0, 4096 environments, PPO batch size 16384, one GPU per run,
786432000 frames (6000 epochs), evaluation every 200 epochs, shape seed 42.
Dataset: /workspace/motion_cache/small150_128shape.pt. Refined data is not required.
W&B: yugoamaryl/hhi-protomotions, group hhi_150_evalfix_comparison.
Run names: hhi_150_evalfix_unrefined_{F,G,H}_seed0.
Logs: /tmp/hhi_150_evalfix_unrefined/; checkpoints: results/<run name>/.

Launch on RunPod with the training environment activated:
  nohup bash tools/train_150_evalfix_unrefined.sh \
    > /tmp/hhi_150_evalfix_unrefined_launcher.log 2>&1 &

--dry-run prints commands without requiring GPUs/data or creating files.
--all-seeds also queues seed 1 for F/G/H after each GPU's seed-0 run.
Use ABLATION_GPUS=0,1 to queue the three runs on only two GPUs if necessary.
Select unused GPUs; this launcher does not inspect or stop unrelated GPU jobs.

Overrides inherited from train_150_evalfix_ablation.sh:
  ABLATION_GPUS       Comma-separated GPU indices (default: 0,1,2)
  ABLATION_RUNS       CASE:SEED pairs, restricted to F/G/H (default: F:0 G:0 H:0)
  ABLATION_PREFIX     Run prefix (default: hhi_150_evalfix_unrefined)
  ABLATION_LOG_DIR    Log directory (default: /tmp/hhi_150_evalfix_unrefined)
  ABLATION_PORT_BASE  Per-GPU port base (default: 29700)
  ABLATION_MOTION_DIR, ABLATION_PYTHON, ABLATION_NUM_ENVS,
  ABLATION_BATCH_SIZE, ABLATION_MAX_STEPS retain the shared launcher's defaults.

Existing checkpoints resume, and logs are appended. Keep settings unchanged
when resuming; use a new prefix for a fresh comparison. Separate run names
prevent accidentally resuming A/B/D's refined-data checkpoints.
Compare final models on common reference targets, not just their own training targets.
USAGE
}

default_runs='F:0 G:0 H:0'
forward_args=()
for option in "$@"; do
    case "$option" in
        --dry-run) forward_args+=(--dry-run) ;;
        --all-seeds) default_runs='F:0 G:0 H:0 F:1 G:1 H:1' ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

export ABLATION_GPUS="${ABLATION_GPUS:-0,1,2}"
export ABLATION_RUNS="${ABLATION_RUNS:-$default_runs}"
export ABLATION_PREFIX="${ABLATION_PREFIX:-hhi_150_evalfix_unrefined}"
export ABLATION_LOG_DIR="${ABLATION_LOG_DIR:-/tmp/hhi_150_evalfix_unrefined}"
export ABLATION_PORT_BASE="${ABLATION_PORT_BASE:-29700}"

# Reject inherited A-E selections rather than silently launch the wrong data condition.
read -r -a unrefined_specs <<< "$ABLATION_RUNS"
for run_spec in "${unrefined_specs[@]}"; do
    if [[ ! "$run_spec" =~ ^[F-H]:(0|[1-9][0-9]*)$ ]]; then
        printf 'ERROR: Unrefined launcher accepts only F/G/H CASE:SEED pairs, got: %s\n' "$run_spec" >&2
        exit 2
    fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# exec preserves the shared launcher's lock, queue, failure, and signal behavior.
exec bash "${script_dir}/train_150_evalfix_ablation.sh" "${forward_args[@]}"
