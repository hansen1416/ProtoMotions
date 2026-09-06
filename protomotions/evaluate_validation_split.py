# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Evaluate a checkpoint on the frozen GlobalClipPool validation split.

Validation clips are streamed in bounded clip batches. Each per-clip file contains all body
shapes, but only one deterministic rotating shape is simulated for each clip, using an
environment with the identical morphology asset. This reproduces the corrected training-time
``eval_holdout/*`` protocol without materializing the complete validation corpus on one GPU.

The script downloads manifests and missing clip files from the checkpoint's configured R2 source.
Processed clip files are removed from the script-owned cache by default, so peak disk usage is
approximately one clip batch. It writes an incrementally checkpointed summary JSON plus a JSONL
record for every evaluated clip-shape pair.

Only validation can be opened here. The test manifest intentionally remains inaccessible through
GlobalClipPool; final test evaluation is a separate one-way gate.

Usage:
    python protomotions/evaluate_validation_split.py \
        --checkpoint results/<experiment>/last.ckpt \
        --simulator isaacgym \
        --num-envs 128 \
        --clip-batch-size 128
"""


def create_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream a checkpoint over the frozen validation split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--simulator", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--clip-batch-size", type=int, default=128)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Motion download cache (default: <checkpoint dir>/validation_motion_cache)",
    )
    parser.add_argument(
        "--keep-downloaded-clips",
        action="store_true",
        help="Keep processed validation .pt files instead of bounding disk use per batch",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=None,
        help="Evaluate only the first N validation clips (smoke testing only)",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Summary JSON (default: <checkpoint dir>/validation_eval_report.json)",
    )
    parser.add_argument(
        "--details-output",
        type=str,
        default=None,
        help="Per-motion JSONL (default: <summary stem>_per_motion.jsonl)",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in key=value format",
    )
    return parser


import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.agents.evaluators.mimic_evaluator import (  # noqa: E402
    MOTION_SOURCE_NAMES,
)
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def _finite_float(value: Any) -> Optional[float]:
    number = float(value)
    return number if math.isfinite(number) else None


def _metadata_value(values: Any, index: int) -> Optional[str]:
    if values is None:
        return None
    value = values[index]
    return None if value is None else str(value)


def _extract_per_motion_records(evaluator, batch_index: int) -> List[Dict[str, Any]]:
    """Capture analyzable per-motion results before evaluator buffers are released."""
    motion_lib = evaluator.motion_lib
    selected = evaluator._eval_motion_subset
    if selected is None:
        selected = torch.arange(
            motion_lib.num_motions(), dtype=torch.long, device=evaluator.device
        )
    selected_cpu = selected.detach().cpu().tolist()
    num_selected = len(selected_cpu)
    if evaluator._motion_failed is None or evaluator._motion_failed.numel() != num_selected:
        raise RuntimeError("Evaluation failure buffer does not match the selected motion panel.")

    overall_failed = evaluator._motion_failed.detach().cpu()
    source_ids = None
    if motion_lib.motion_source_id is not None:
        source_ids = motion_lib.motion_source_id[selected].detach().cpu().tolist()

    component_data: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in evaluator.config.evaluation_components:
        component_data[name] = {
            "failed": evaluator._per_component_failures[name].detach().cpu(),
            "sum": evaluator._component_value_sum[name].detach().cpu(),
            "min": evaluator._component_value_min[name].detach().cpu(),
            "max": evaluator._component_value_max[name].detach().cpu(),
            "count": evaluator._component_step_count[name].detach().cpu(),
        }

    normalized_jerk: List[Optional[float]] = [None] * num_selected
    high_jerk_percentage: List[Optional[float]] = [None] * num_selected
    smoothness_plugin = next(
        (
            plugin
            for plugin in evaluator.metric_plugins
            if hasattr(plugin, "smoothness_calculator")
        ),
        None,
    )
    if smoothness_plugin is not None and "rigid_body_pos" in evaluator._metrics:
        calculator = smoothness_plugin.smoothness_calculator
        per_motion_nj, _, windowed_nj = calculator.compute_normalized_jerk_from_pos(
            evaluator._metrics["rigid_body_pos"], smoothness_plugin.num_bodies
        )
        for index in range(num_selected):
            value = float(per_motion_nj[index].item())
            if value > 0 and math.isfinite(value):
                normalized_jerk[index] = value
            windows = windowed_nj[index]
            if windows.numel() > 0:
                high_jerk_percentage[index] = calculator._compute_high_jerk_frame_percentage(
                    windows
                )

    action_values: List[Dict[str, Optional[float]]] = [dict() for _ in selected_cpu]
    actions_metric = evaluator._metrics.get("actions")
    if actions_metric is not None:
        for index in range(num_selected):
            frame_count = int(actions_metric.frame_counts[index].item())
            if frame_count < 2:
                continue
            actions = actions_metric.data[index, :frame_count]
            deltas = (actions[1:] - actions[:-1]).abs()
            mean_delta = float(deltas.mean().item())
            mean_max_delta = float(deltas.max(dim=-1).values.mean().item())
            action_values[index] = {
                "action_delta_mean_rad": mean_delta,
                "action_delta_max_rad": mean_max_delta,
                "action_rate_mean_rad_s": mean_delta / evaluator.env.dt,
                "action_delta_mean_deg": mean_delta * 180.0 / math.pi,
                "action_delta_max_deg": mean_max_delta * 180.0 / math.pi,
            }

    records: List[Dict[str, Any]] = []
    for local_index, motion_id in enumerate(selected_cpu):
        components = {}
        evaluated_frames = 0
        for name, values in component_data.items():
            count = int(values["count"][local_index].item())
            evaluated_frames = max(evaluated_frames, count)
            components[name] = {
                "failed": bool(values["failed"][local_index].item()),
                "frame_count": count,
                "mean": (
                    _finite_float(values["sum"][local_index].item() / count)
                    if count > 0
                    else None
                ),
                "min": (
                    _finite_float(values["min"][local_index].item())
                    if count > 0
                    else None
                ),
                "max": (
                    _finite_float(values["max"][local_index].item())
                    if count > 0
                    else None
                ),
            }

        source_id = None if source_ids is None else int(source_ids[local_index])
        record = {
            "batch_index": batch_index,
            "motion_id_in_batch_library": int(motion_id),
            "clip_id": _metadata_value(motion_lib.motion_clip_ids, motion_id),
            "base_clip_id": _metadata_value(
                getattr(motion_lib, "motion_base_clip_ids", None), motion_id
            ),
            "asset_id": _metadata_value(motion_lib.motion_asset_ids, motion_id),
            "beta_key": _metadata_value(motion_lib.motion_beta_keys, motion_id),
            "npz_file": _metadata_value(
                getattr(motion_lib, "motion_npz_files", None), motion_id
            ),
            "source_id": source_id,
            "source_name": MOTION_SOURCE_NAMES.get(source_id),
            "motion_length_seconds": _finite_float(
                motion_lib.motion_lengths[motion_id].item()
            ),
            "evaluated_frames": evaluated_frames,
            "failed": bool(overall_failed[local_index].item()),
            "components": components,
            "normalized_jerk": normalized_jerk[local_index],
            "high_jerk_frame_percentage": high_jerk_percentage[local_index],
            **action_values[local_index],
        }
        records.append(record)
    return records


def _mean_present(records: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [record.get(key) for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def _aggregate_records(records: Sequence[Dict[str, Any]], evaluator_config) -> Dict[str, Any]:
    """Aggregate per-motion records using the same motion-weighted reductions as evaluation."""
    if not records:
        return {}

    metrics: Dict[str, Any] = {
        "eval_holdout/num_motions": len(records),
        "eval_holdout/success_rate": 1.0
        - sum(record["failed"] for record in records) / len(records),
    }

    for name in evaluator_config.evaluation_components:
        component_rows = [
            record["components"][name]
            for record in records
            if record["components"][name]["frame_count"] > 0
        ]
        if not component_rows:
            continue
        component_config = evaluator_config.evaluation_components[name]
        if component_config.static_params.get("threshold") is not None:
            metrics[f"eval_holdout/{name}/failure_rate"] = sum(
                row["failed"] for row in component_rows
            ) / len(component_rows)
        metrics[f"eval_holdout/{name}/mean"] = sum(
            row["mean"] for row in component_rows
        ) / len(component_rows)
        metrics[f"eval_holdout/{name}/max"] = max(
            row["max"] for row in component_rows
        )
        metrics[f"eval_holdout/{name}/min"] = min(
            row["min"] for row in component_rows
        )

    additional_fields = {
        "normalized_jerk": "eval_holdout/normalized_jerk_mean",
        "high_jerk_frame_percentage": "eval_holdout/high_jerk_frame_percentage_mean",
        "action_delta_mean_rad": "eval_holdout/action_delta_mean_rad",
        "action_delta_max_rad": "eval_holdout/action_delta_max_rad",
        "action_rate_mean_rad_s": "eval_holdout/action_rate_mean_rad_s",
        "action_delta_mean_deg": "eval_holdout/action_delta_mean_deg",
        "action_delta_max_deg": "eval_holdout/action_delta_max_deg",
    }
    for record_key, metric_key in additional_fields.items():
        value = _mean_present(records, record_key)
        if value is not None:
            metrics[metric_key] = value

    known_source_records: List[Dict[str, Any]] = []
    known_source_failures = 0
    source_success_components = getattr(
        evaluator_config, "source_success_components", {}
    ) or {}
    for source_id, source_name in MOTION_SOURCE_NAMES.items():
        source_records = [
            record for record in records if record["source_id"] == source_id
        ]
        if not source_records:
            continue
        configured_components = source_success_components.get(source_id)
        if configured_components is None:
            configured_components = source_success_components.get(str(source_id))

        def source_failed(record):
            if configured_components:
                return any(
                    record["components"][name]["failed"]
                    for name in configured_components
                )
            return record["failed"]

        failures = sum(source_failed(record) for record in source_records)
        known_source_records.extend(source_records)
        known_source_failures += failures
        prefix = f"eval_holdout_{source_name}"
        metrics[f"{prefix}/num_motions"] = len(source_records)
        metrics[f"{prefix}/success_rate"] = 1.0 - failures / len(source_records)
        for name in evaluator_config.evaluation_components:
            component_rows = [record["components"][name] for record in source_records]
            component_config = evaluator_config.evaluation_components[name]
            if component_config.static_params.get("threshold") is not None:
                metrics[f"{prefix}/{name}/failure_rate"] = sum(
                    row["failed"] for row in component_rows
                ) / len(component_rows)
            valid = [row for row in component_rows if row["frame_count"] > 0]
            if valid:
                metrics[f"{prefix}/{name}/mean"] = sum(
                    row["mean"] for row in valid
                ) / len(valid)
                metrics[f"{prefix}/{name}/max"] = max(row["max"] for row in valid)
                metrics[f"{prefix}/{name}/min"] = min(row["min"] for row in valid)

    if known_source_records:
        metrics["eval_holdout/source_conditioned_success_rate"] = (
            1.0 - known_source_failures / len(known_source_records)
        )
    return metrics


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _remove_cached_clip_files(motion_lib, clip_ids: Sequence[str], remote_map) -> None:
    for clip_id in clip_ids:
        remote_name = remote_map.get(clip_id)
        if remote_name is None:
            continue
        cached_path = motion_lib.local_cache_dir / remote_name
        if cached_path.is_file():
            cached_path.unlink()


def _evaluate_clip_batch(agent, motion_lib, clip_ids, batch_index):
    evaluator = agent.evaluator
    motion_lib.load_eval_holdout(clip_ids=clip_ids)
    agent.env.motion_manager.on_motion_lib_reloaded()
    all_env_ids = torch.arange(agent.env.num_envs, device=evaluator.device)
    agent.env.reset(all_env_ids)

    previous_skip_weight_update = evaluator._skip_weight_update
    previous_save_predicted = evaluator.config.save_predicted_motion_lib_every
    evaluator._skip_weight_update = True
    evaluator.config.save_predicted_motion_lib_every = None
    initialized = False
    try:
        evaluator._metrics = evaluator.initialize_eval()
        initialized = True
        if evaluator._metrics is None:
            raise RuntimeError("Evaluator returned no metric buffers.")
        evaluator.run_evaluation()
        records = _extract_per_motion_records(evaluator, batch_index)
        chunk_log, _ = evaluator.process_eval_results()
    finally:
        evaluator._skip_weight_update = previous_skip_weight_update
        evaluator.config.save_predicted_motion_lib_every = previous_save_predicted
        if initialized:
            evaluator.cleanup_after_evaluation()

    holdout_chunk_log = {}
    for key, value in chunk_log.items():
        if key.startswith("eval/"):
            key = key.replace("eval/", "eval_holdout/", 1)
        elif key.startswith("eval_"):
            key = key.replace("eval_", "eval_holdout_", 1)
        holdout_chunk_log[key] = float(value)
    return records, holdout_chunk_log


def main():
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.clip_batch_size <= 0:
        parser.error("--clip-batch-size must be positive")
    if args.limit_clips is not None and args.limit_clips <= 0:
        parser.error("--limit-clips must be positive")

    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    if not resolved_configs_path.exists():
        raise FileNotFoundError(f"Could not find resolved configs at {resolved_configs_path}")

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' to '{args.simulator}'"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)
    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless

    if "global_clip_pool" not in motion_lib_config._target_:
        raise RuntimeError(
            f"This checkpoint's motion_lib is {motion_lib_config._target_!r}, not "
            "GlobalClipPool -- there is no frozen validation split to evaluate."
        )
    if motion_lib_config.validation_manifest_name is None:
        raise RuntimeError(
            "motion_lib.validation_manifest_name is not set -- this run did not use "
            "an explicit train/validation split."
        )

    from protomotions.utils.config_utils import apply_config_overrides, parse_cli_overrides

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None
    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    # Standalone validation does not need a meaningful training resident pool. Keeping one clip
    # makes construction cheap and prevents the stale inference config's 256x3 cache settings
    # from recreating the training-time OOM/disk footprint.
    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir is not None
        else checkpoint.parent / "validation_motion_cache"
    )
    motion_lib_config.local_cache_dir = str(cache_dir)
    motion_lib_config.resident_pool_size = 1
    motion_lib_config.cache_size_multiplier = 1.0

    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = getattr(env_config, "save_dir", None)
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(checkpoint, load_env=False)
    agent.eval()

    motion_lib = components["motion_lib"]
    if not motion_lib.has_eval_holdout():
        raise RuntimeError(
            "GlobalClipPool reports no validation holdout -- check the saved manifest config."
        )

    validation_clip_ids = list(motion_lib.eval_holdout_clip_ids)
    if args.limit_clips is not None:
        validation_clip_ids = validation_clip_ids[: args.limit_clips]
    num_batches = math.ceil(len(validation_clip_ids) / args.clip_batch_size)
    shape_panel_index = agent.evaluator._eval_shape_panel_index()

    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else checkpoint.parent / "validation_eval_report.json"
    )
    details_path = (
        Path(args.details_output).resolve()
        if args.details_output is not None
        else output_path.with_name(output_path.stem + "_per_motion.jsonl")
    )
    if output_path == details_path:
        raise ValueError("--output and --details-output must name different files.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        f"Evaluating epoch {agent.current_epoch} on {len(validation_clip_ids)} validation "
        f"clips in {num_batches} batches of at most {args.clip_batch_size}; "
        f"shape_panel_index={shape_panel_index}, cache={cache_dir}."
    )

    all_records: List[Dict[str, Any]] = []
    batch_reports: List[Dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    def current_report(complete: bool) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "complete": complete,
            "started_at_utc": started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "epoch": int(agent.current_epoch),
            "step_count": int(agent.step_count),
            "split_provenance": motion_lib.split_provenance(),
            "shape_panel_index": int(shape_panel_index),
            "num_validation_clips_expected": len(validation_clip_ids),
            "num_validation_clips_processed": len(all_records),
            "clip_batch_size": int(args.clip_batch_size),
            "num_batches_expected": num_batches,
            "num_batches_completed": len(batch_reports),
            "num_envs": int(args.num_envs),
            "details_file": str(details_path),
            "metrics": _aggregate_records(all_records, agent.evaluator.config),
            "batches": batch_reports,
        }

    try:
        with open(details_path, "w") as details_stream:
            for batch_index, start in enumerate(
                range(0, len(validation_clip_ids), args.clip_batch_size)
            ):
                clip_ids = validation_clip_ids[start : start + args.clip_batch_size]
                log.info(
                    f"Validation clip batch {batch_index + 1}/{num_batches}: "
                    f"{len(clip_ids)} clips"
                )
                records, evaluator_chunk_metrics = _evaluate_clip_batch(
                    agent, motion_lib, clip_ids, batch_index
                )
                if len(records) != len(clip_ids):
                    raise RuntimeError(
                        f"Batch {batch_index} selected {len(records)} motions for "
                        f"{len(clip_ids)} clips; expected exactly one shape per clip."
                    )
                for record in records:
                    details_stream.write(json.dumps(record, sort_keys=True) + "\n")
                details_stream.flush()
                os.fsync(details_stream.fileno())

                batch_metrics = _aggregate_records(records, agent.evaluator.config)
                observed_success = evaluator_chunk_metrics.get(
                    "eval_holdout/success_rate"
                )
                recorded_success = batch_metrics["eval_holdout/success_rate"]
                if observed_success is None or not math.isclose(
                    observed_success, recorded_success, rel_tol=0.0, abs_tol=1e-6
                ):
                    raise RuntimeError(
                        "Per-motion records do not reproduce the evaluator's batch success "
                        f"rate: records={recorded_success}, evaluator={observed_success}."
                    )

                all_records.extend(records)
                batch_reports.append(
                    {
                        "batch_index": batch_index,
                        "num_clips": len(clip_ids),
                        "first_clip_id": clip_ids[0],
                        "last_clip_id": clip_ids[-1],
                        "metrics": batch_metrics,
                        "evaluator_metrics": evaluator_chunk_metrics,
                    }
                )
                _write_json_atomic(output_path, current_report(complete=False))

                if not args.keep_downloaded_clips:
                    _remove_cached_clip_files(
                        motion_lib, clip_ids, motion_lib._eval_holdout_remote_name
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if not args.keep_downloaded_clips:
            resident_indices = motion_lib._resident_local_indices
            if resident_indices is not None:
                resident_clip_ids = [
                    motion_lib.rank_clip_ids[index]
                    for index in resident_indices.detach().cpu().tolist()
                ]
                _remove_cached_clip_files(
                    motion_lib, resident_clip_ids, motion_lib._clip_remote_name
                )
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()

    report = current_report(complete=True)
    _write_json_atomic(output_path, report)

    print("\n" + "=" * 68)
    print(
        f"VALIDATION SPLIT EVALUATION ({motion_lib.config.split_version}, "
        f"panel {shape_panel_index})"
    )
    print("=" * 68)
    for key, value in sorted(report["metrics"].items()):
        print(f"  {key}: {value}")
    print("=" * 68)
    print(f"Summary:    {output_path}")
    print(f"Per motion: {details_path}\n")
    log.info(f"Wrote complete validation report to {output_path}")


if __name__ == "__main__":
    main()
