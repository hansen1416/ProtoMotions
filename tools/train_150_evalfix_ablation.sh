#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
# Run independent architecture/data conditions on six GPUs, with one job per GPU.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash tools/train_150_evalfix_ablation.sh [--dry-run] [--all-seeds]

Conditions: A=temporal MLP, B=attention, C=slot/type, D=slot/type+AdaLN,
            E=slot/type with unrefined HUMOS. A-D use refined HUMOS.
Default six jobs: A:0 B:0 C:0 D:0 E:0 C:1, on GPUs 0 through 5 respectively.
--all-seeds adds A:1 B:1 D:1 E:1 to complete the original ten-run comparison.
Additional jobs queue round-robin by GPU, never concurrently on the same GPU.
A failed run stops only its GPU's queue.
Existing checkpoints resume through train_agent.py; logs are appended.

Start on RunPod from the repository with the training environment activated:
  nohup bash tools/train_150_evalfix_ablation.sh \
    > /tmp/hhi_150_evalfix_launcher.log 2>&1 &

Optional environment settings:
  ABLATION_GPUS              Comma-separated GPU indices (default: 0,1,2,3,4,5)
  ABLATION_RUNS              Space-separated CASE:SEED pairs; overrides the run list
  ABLATION_MOTION_DIR        Motion directory (default: /workspace/motion_cache)
  ABLATION_LOG_DIR           Per-run log directory (default: /tmp/hhi_150_evalfix)
  ABLATION_PREFIX            Run-name prefix (default: hhi_150_evalfix)
  ABLATION_PYTHON            Python executable (default: python)
  ABLATION_NUM_ENVS          Environments per GPU (default: 4096)
  ABLATION_BATCH_SIZE        PPO batch size per run (default: 16384)
  ABLATION_MAX_STEPS         Training frames per run (default: 786432000)
  ABLATION_PORT_BASE         First distributed port; one per GPU (default: 29600)

Defaults give 6,000 epochs per run (4096 envs x 32 steps x 6000 epochs).
Keep settings identical when resuming: saved configs override training CLI flags.
Use a new ABLATION_PREFIX for a fresh comparison with changed settings.
Dry-run prints the selected commands without requiring GPUs/data or creating files.
E's training-time evaluation uses raw targets; the final C/E paper comparison
needs a separate evaluation against common reference motions.
USAGE
}

dry_run=false
default_runs='A:0 B:0 C:0 D:0 E:0 C:1'
for option in "$@"; do
    case "$option" in
        --dry-run) dry_run=true ;;
        --all-seeds) default_runs='A:0 B:0 C:0 D:0 E:0 C:1 A:1 B:1 D:1 E:1' ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd -- "$repo_root"

gpu_list="${ABLATION_GPUS:-0,1,2,3,4,5}"
read -r -a run_specs <<< "${ABLATION_RUNS:-$default_runs}"
motion_dir="${ABLATION_MOTION_DIR:-/workspace/motion_cache}"
log_dir="${ABLATION_LOG_DIR:-/tmp/hhi_150_evalfix}"
run_prefix="${ABLATION_PREFIX:-hhi_150_evalfix}"
python_bin="${ABLATION_PYTHON:-python}"
num_envs="${ABLATION_NUM_ENVS:-4096}"
batch_size="${ABLATION_BATCH_SIZE:-16384}"
max_steps="${ABLATION_MAX_STEPS:-786432000}"
port_base="${ABLATION_PORT_BASE:-29600}"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$gpu_list" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail 'ABLATION_GPUS must contain comma-separated GPU indices.'
IFS=',' read -r -a gpu_ids <<< "$gpu_list"
declare -A seen_gpus=() seen_runs=()
for gpu in "${gpu_ids[@]}"; do
    [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ ]] || fail "Invalid GPU index: $gpu"
    [[ -z "${seen_gpus[$gpu]:-}" ]] || fail "GPU $gpu was assigned twice."
    seen_gpus[$gpu]=1
done
(( ${#run_specs[@]} > 0 )) || fail 'At least one CASE:SEED pair is required.'
for run_spec in "${run_specs[@]}"; do
    [[ "$run_spec" =~ ^[A-E]:(0|[1-9][0-9]*)$ ]] || fail "Invalid CASE:SEED pair: $run_spec"
    [[ -z "${seen_runs[$run_spec]:-}" ]] || fail "Run $run_spec was supplied twice."
    seen_runs[$run_spec]=1
done
[[ "$run_prefix" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail 'Invalid ABLATION_PREFIX.'
for value in "$num_envs" "$batch_size" "$max_steps" "$port_base"; do
    [[ "$value" =~ ^[1-9][0-9]*$ && ${#value} -le 12 ]] || fail "Invalid positive integer: $value"
done
(( port_base >= 1024 && port_base + ${#gpu_ids[@]} - 1 <= 65535 )) || fail 'Per-GPU ports must be between 1024 and 65535.'
(( max_steps >= num_envs * 32 )) || fail 'Frame budget must allow at least one epoch.'
(( max_steps % (num_envs * 32) == 0 )) || fail 'Frame budget must be divisible by num_envs x 32.'

declare -A experiments=(
    [A]=mlp_wide_discover_historical_lookahead
    [B]=mlp_wide_discover_attention
    [C]=mlp_wide_discover_attention_slot_type
    [D]=mlp_wide_discover_attention_adaln
    [E]=mlp_wide_discover_attention_slot_type
)
refined_motion="${motion_dir}/small150_128shape_refined.pt"
raw_motion="${motion_dir}/small150_128shape.pt"

if ! "$dry_run"; then
    for dependency in "$python_bin" nvidia-smi flock setsid; do
        command -v "$dependency" >/dev/null || fail "Required command not found: $dependency"
    done
    for motion_file in "$refined_motion" "$raw_motion"; do
        [[ -r "$motion_file" && -s "$motion_file" ]] || fail "Motion file missing, empty or unreadable: $motion_file"
    done
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" --query-gpu=index,name --format=csv,noheader || fail "GPU $gpu is unavailable."
    done
    mkdir -p -- "$log_dir" "${repo_root}/results"
    # The kernel releases this lock automatically when this launcher and its workers exit.
    exec 9>"${repo_root}/results/.${run_prefix}.launcher.lock"
    flock -n 9 || fail "A launcher for $run_prefix is already active in this repository."
fi

log() { printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"; }

run_lane() {
    local lane="$1" gpu port case_id child_pid job run_spec seed experiment motion_file
    local run_name log_file exit_code
    local -a command
    gpu="${gpu_ids[$lane]}"
    port="$((port_base + lane))"
    case_id='pending'
    child_pid=""

    stop_lane() {
        trap '' INT TERM
        if [[ -n "$child_pid" ]]; then
            log "STOP case=$case_id GPU=$gpu training_pid=$child_pid"
            # setsid gives each training job its own group, including child processes.
            kill -TERM -- "-$child_pid" 2>/dev/null || true
            wait "$child_pid" 2>/dev/null || true
        fi
        exit 130
    }
    trap stop_lane INT TERM

    for (( job=lane; job<${#run_specs[@]}; job+=${#gpu_ids[@]} )); do
        run_spec="${run_specs[$job]}"
        case_id="${run_spec%%:*}"
        seed="${run_spec#*:}"
        experiment="examples/experiments/mimic/${experiments[$case_id]}.py"
        motion_file="$refined_motion"
        [[ "$case_id" != E ]] || motion_file="$raw_motion"
        [[ -f "$experiment" ]] || fail "Experiment not found: $experiment"
        run_name="${run_prefix}_${case_id}_seed${seed}"
        log_file="${log_dir}/${run_name}.log"
        command=(
            "$python_bin" -u protomotions/train_agent.py
            --robot-name smpl_mor --simulator isaacgym
            --experiment-path "$experiment"
            --experiment-name "$run_name"
            --motion-file "$motion_file"
            --num-envs "$num_envs" --batch-size "$batch_size" --ngpu 1
            --seed "$seed" --training-max-steps "$max_steps"
            --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl
            --wandb-group hhi_150_evalfix_comparison
            --overrides
            agent.evaluator.eval_one_shape_per_motion=True
            agent.evaluator.eval_shape_sampling_seed=42
            agent.evaluator.eval_metrics_every=200
        )

        if "$dry_run"; then
            printf 'CUDA_VISIBLE_DEVICES=%q MASTER_ADDR=127.0.0.1 MASTER_PORT=%q ' "$gpu" "$port"
            printf '%q ' "${command[@]}"
            printf '>> %q 2>&1\n' "$log_file"
            continue
        fi

        log "START case=$case_id seed=$seed GPU=$gpu port=$port name=$run_name log=$log_file"
        log "START $run_name GPU=$gpu port=$port" >> "$log_file"
        printf '%q ' "${command[@]}" >> "$log_file"
        printf '\n' >> "$log_file"
        # nohup belongs on the outer launcher; wait before using this GPU for another run.
        CUDA_VISIBLE_DEVICES="$gpu" MASTER_ADDR=127.0.0.1 MASTER_PORT="$port" \
            setsid "${command[@]}" >> "$log_file" 2>&1 &
        child_pid=$!
        if wait "$child_pid"; then
            child_pid=""
            log "DONE case=$case_id seed=$seed GPU=$gpu name=$run_name"
        else
            exit_code=$?
            child_pid=""
            log "FAILED case=$case_id seed=$seed GPU=$gpu exit=$exit_code; queue stopped, see $log_file"
            exit "$exit_code"
        fi
    done
}

if "$dry_run"; then
    for lane in "${!gpu_ids[@]}"; do
        (run_lane "$lane")
    done
    exit 0
fi

worker_pids=()
stop_all() {
    trap '' INT TERM
    log 'Stopping all GPU queues.'
    for pid in "${worker_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${worker_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_all INT TERM

log "Launching ${#gpu_ids[@]} GPU queues; runs=${run_specs[*]}; frames/run=$max_steps; epochs/run=$((max_steps / num_envs / 32))"
for lane in "${!gpu_ids[@]}"; do
    run_lane "$lane" &
    worker_pids+=("$!")
done

failed_lanes=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
        failed_lanes=$((failed_lanes + 1))
    fi
done
if (( failed_lanes > 0 )); then
    fail "$failed_lanes GPU queue(s) failed. Other queues have finished; inspect per-run logs."
fi
log 'All requested runs exited successfully.'
