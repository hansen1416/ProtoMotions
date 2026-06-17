"""Simple HHI fall detector.

Run each morphology/motion entry, compare simulated rigid-body positions with
the target rigid-body positions, and rank templates by mean distance.

No threshold. No gt_error. No reset/termination logic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.mimic_evaluator import (
    MimicEvaluator,
    MimicEpisodeContext,
)

MorphKey = Tuple[str, str]  # (gender, beta_key)


class HHIFaultEvaluator(MimicEvaluator):
    """Minimal morphology evaluator based only on rollout-target distance.

    Important invariant:
        each env must evaluate only motions with the same (gender, beta_key)
        as the humanoid asset assigned to that env.
    """

    def __init__(
        self,
        agent: Any,
        fabric: Any,
        config: Any,
        output_path: Optional[Path] = None,
        progress_every: int = 50,
    ):
        super().__init__(agent=agent, fabric=fabric, config=config)

        self.output_path = Path(output_path) if output_path is not None else None
        self.progress_every = progress_every

    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict, Optional[float]]:
        """Override BaseEvaluator.evaluate so this works without evaluation_components."""
        self.agent.eval()

        self._metrics = self.initialize_eval()
        self.run_evaluation()
        evaluation_log, evaluated_score = self.process_eval_results()
        self.cleanup_after_evaluation()
        self.eval_count += 1

        return evaluation_log, evaluated_score

    def initialize_eval(self) -> Dict:
        num_motions = self.motion_lib.num_motions()

        self._env_snapshot = self.env.save_state()
        self._cached_motion_ids = self.motion_manager.motion_ids.clone()
        self._cached_motion_times = self.motion_manager.motion_times.clone()

        self._count = torch.zeros(num_motions, dtype=torch.long, device=self.device)

        self._sum_body_dist = torch.zeros(
            num_motions, dtype=torch.float, device=self.device
        )
        self._max_body_dist = torch.zeros(
            num_motions, dtype=torch.float, device=self.device
        )

        self._sum_root_dist = torch.zeros(
            num_motions, dtype=torch.float, device=self.device
        )
        self._max_root_dist = torch.zeros(
            num_motions, dtype=torch.float, device=self.device
        )

        self._min_root_height = torch.full(
            (num_motions,), float("inf"), dtype=torch.float, device=self.device
        )

        # motion_id -> last env_id used to evaluate it. This is only for reporting.
        self._matched_env_id = torch.full(
            (num_motions,), -1, dtype=torch.long, device=self.device
        )

        return {}

    def run_evaluation(self) -> None:
        batches = self._build_gender_beta_matched_batches()

        for batch_idx, (env_ids, motion_ids) in enumerate(batches, start=1):
            self._assert_gender_beta_consistency(env_ids, motion_ids)

            motion_lengths = self.motion_lib.get_motion_length(motion_ids)

            # Evaluate the full motion. The loop count is derived only from motion length.
            frame_limits = (motion_lengths / self.env.dt).floor().long()
            num_frames = int(frame_limits.max().item())

            motion_start = int(motion_ids.min().item())
            motion_end = int(motion_ids.max().item())

            print(
                f"Running batch {batch_idx}/{len(batches)}: "
                f"{motion_ids.numel()} matched motions, "
                f"motion_id range {motion_start}..{motion_end}, "
                f"frames={num_frames}",
                flush=True,
            )

            self._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                local_ids=motion_ids,
                frame_limits=frame_limits,
            )

            self.evaluate_episode(env_ids)

    def evaluate_episode(self, env_ids: Tensor) -> None:
        self._on_episode_start(env_ids)

        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        ema_alpha = self.config.eval_action_ema_alpha
        prev_actions = None

        num_frames = int(self._episode_ctx.frame_limits.max().item())

        for frame_idx in range(num_frames):
            if self.progress_every > 0 and frame_idx % self.progress_every == 0:
                print(f"  frame {frame_idx}/{num_frames}", flush=True)

            self._record_distance(env_ids, frame_idx)

            if frame_idx == num_frames - 1:
                break

            model_outs = self.agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))

            if ema_alpha is not None:
                if prev_actions is None:
                    prev_actions = actions.clone()
                actions = ema_alpha * actions + (1.0 - ema_alpha) * prev_actions
                prev_actions = actions.clone()

            obs, rewards, dones, terminated, extras = self.env.step(actions)

            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

    def _build_gender_beta_matched_batches(self) -> List[Tuple[Tensor, Tensor]]:
        """Build batches of (env_ids, motion_ids) with matching (gender, beta_key).

        The simulator's humanoid asset is fixed after IsaacGym env creation, so this
        evaluator does not try to change an env's body shape. Instead, it selects an
        already-created env with a matching morphology for each motion.
        """
        env_key_to_env_ids = self._build_env_morph_key_to_env_ids()

        motion_key_to_motion_ids: Dict[MorphKey, List[int]] = {}
        for motion_id in range(self.motion_lib.num_motions()):
            motion_key = self._get_motion_morph_key(motion_id)
            motion_key_to_motion_ids.setdefault(motion_key, []).append(motion_id)

        missing_keys = sorted(
            key for key in motion_key_to_motion_ids if key not in env_key_to_env_ids
        )
        if missing_keys:
            available = sorted(env_key_to_env_ids.keys())
            raise RuntimeError(
                "Some motions have no humanoid env with the same (gender, beta_key). "
                "The evaluator cannot change IsaacGym humanoid assets after env creation. "
                "Use enough envs, or create envs with requested morphology asset ids. "
                f"Missing keys examples: {missing_keys[:10]}. "
                f"Available env keys examples: {available[:10]}."
            )

        # Greedy packing. One env can evaluate at most one motion in a batch,
        # but it can be reused in later batches.
        queues = {key: ids[:] for key, ids in motion_key_to_motion_ids.items()}
        batches: List[Tuple[Tensor, Tensor]] = []

        while any(len(ids) > 0 for ids in queues.values()):
            used_env_ids = set()
            batch_pairs: List[Tuple[int, int]] = []

            for key in sorted(queues.keys()):
                motion_queue = queues[key]
                if len(motion_queue) == 0:
                    continue

                for env_id in env_key_to_env_ids[key]:
                    if len(motion_queue) == 0:
                        break
                    if env_id in used_env_ids:
                        continue

                    motion_id = motion_queue.pop(0)
                    batch_pairs.append((env_id, motion_id))
                    used_env_ids.add(env_id)
                    self._matched_env_id[motion_id] = env_id

            if len(batch_pairs) == 0:
                raise RuntimeError(
                    "Failed to build a non-empty gender-beta matched evaluation batch. "
                    "This usually means no available env can match the remaining motions."
                )

            env_ids = torch.tensor(
                [env_id for env_id, _ in batch_pairs],
                device=self.device,
                dtype=torch.long,
            )
            motion_ids = torch.tensor(
                [motion_id for _, motion_id in batch_pairs],
                device=self.device,
                dtype=torch.long,
            )
            batches.append((env_ids, motion_ids))

        print(
            f"Built {len(batches)} gender-beta matched evaluation batches "
            f"for {self.motion_lib.num_motions()} motions using {self.num_envs} envs.",
            flush=True,
        )

        return batches

    def _build_env_morph_key_to_env_ids(self) -> Dict[MorphKey, List[int]]:
        mapping: Dict[MorphKey, List[int]] = {}

        for env_id in range(self.num_envs):
            env_key = self._get_env_morph_key(env_id)
            mapping.setdefault(env_key, []).append(env_id)

        if len(mapping) == 0:
            raise RuntimeError("No humanoid morphology keys found for envs.")

        print(
            "Available humanoid env gender-beta keys: "
            f"{list(mapping.keys())[:10]}",
            flush=True,
        )

        return mapping

    def _assert_gender_beta_consistency(
        self,
        env_ids: Tensor,
        motion_ids: Tensor,
    ) -> None:
        for env_id, motion_id in zip(env_ids.tolist(), motion_ids.tolist()):
            env_key = self._get_env_morph_key(int(env_id))
            motion_key = self._get_motion_morph_key(int(motion_id))

            if env_key != motion_key:
                raise RuntimeError(
                    "Gender-beta mismatch before reset: "
                    f"env_id={env_id}, env_key={env_key}, "
                    f"motion_id={motion_id}, motion_key={motion_key}"
                )

    def _get_env_morph_key(self, env_id: int) -> MorphKey:
        simulator = self.env.simulator

        if not getattr(simulator, "morphology", False):
            raise RuntimeError(
                "HHIFaultEvaluator requires a morphology simulator, but "
                "env.simulator.morphology is False."
            )

        if not hasattr(simulator, "env_id_to_asset_info"):
            raise RuntimeError(
                "Simulator does not expose env_id_to_asset_info. "
                "Cannot check humanoid gender-beta consistency."
            )

        asset_info = simulator.env_id_to_asset_info[int(env_id)]

        try:
            gender = str(asset_info["gender"])
            beta_key = str(asset_info["beta_key"])
        except KeyError as exc:
            raise RuntimeError(
                f"env_id_to_asset_info[{env_id}] does not contain {exc}."
            ) from exc

        if gender == "" or beta_key == "":
            raise RuntimeError(
                f"Empty humanoid morphology key for env_id={env_id}: "
                f"gender={gender!r}, beta_key={beta_key!r}"
            )

        return gender, beta_key

    def _get_motion_morph_key(self, motion_id: int) -> MorphKey:
        _, gender, beta_key = self._get_motion_metadata(motion_id)

        if gender == "" or beta_key == "":
            raise RuntimeError(
                f"Motion {motion_id} does not contain valid gender-beta metadata: "
                f"gender={gender!r}, beta_key={beta_key!r}"
            )

        return gender, beta_key

    def _record_distance(self, env_ids: Tensor, step_idx: int) -> None:
        still_active = self._episode_ctx.frame_limits > step_idx
        if not still_active.any():
            return

        active_env_ids = env_ids[still_active]
        active_motion_ids = self._episode_ctx.motion_ids[still_active]

        pred_state = self.env.simulator.get_robot_state()
        pred_pos = pred_state.rigid_body_pos[active_env_ids]

        motion_times = self.motion_manager.motion_times[active_env_ids]
        target_state = self.motion_lib.get_motion_state(
            active_motion_ids, motion_times
        )
        target_pos = target_state.rigid_body_pos

        # Align target motion to spawned humanoid position.
        target_offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            target_pos, active_env_ids
        )
        target_pos = target_pos + target_offset

        # Mean Euclidean body distance for this frame.
        body_dist = torch.norm(pred_pos - target_pos, dim=-1).mean(dim=-1)

        # Root distance for this frame.
        root_dist = torch.norm(pred_pos[:, 0] - target_pos[:, 0], dim=-1)

        root_height = pred_pos[:, 0, 2]

        self._count[active_motion_ids] += 1

        self._sum_body_dist[active_motion_ids] += body_dist
        self._max_body_dist[active_motion_ids] = torch.maximum(
            self._max_body_dist[active_motion_ids], body_dist
        )

        self._sum_root_dist[active_motion_ids] += root_dist
        self._max_root_dist[active_motion_ids] = torch.maximum(
            self._max_root_dist[active_motion_ids], root_dist
        )

        self._min_root_height[active_motion_ids] = torch.minimum(
            self._min_root_height[active_motion_ids], root_height
        )

    def process_eval_results(self) -> Tuple[Dict, Optional[float]]:
        mean_body_dist = self._sum_body_dist / self._count.clamp(min=1)
        mean_root_dist = self._sum_root_dist / self._count.clamp(min=1)

        rows = self._build_rows(mean_body_dist, mean_root_dist)

        if self.output_path is not None:
            self._save_rows(rows, self.output_path)

        print("\nTop candidates by mean_body_dist:")
        for row in rows[:20]:
            print(
                f"  motion_id={row['motion_id']:>3} "
                f"env_id={row['matched_env_id']:>3} "
                f"gender={row['gender']:<6} "
                f"beta_key={row['beta_key']:<10} "
                f"mean_body_dist={row['mean_body_dist']:.6f} "
                f"max_body_dist={row['max_body_dist']:.6f} "
                f"min_root_height={row['min_root_height']:.6f}",
                flush=True,
            )

        valid = self._count > 0
        if not valid.any():
            raise RuntimeError("No motions were evaluated.")

        to_log = {
            "hhi_fault/mean_body_dist_mean": mean_body_dist[valid].mean().item(),
            "hhi_fault/mean_body_dist_max": mean_body_dist[valid].max().item(),
            "hhi_fault/max_body_dist_max": self._max_body_dist[valid].max().item(),
        }

        # Lower is better.
        score = -mean_body_dist[valid].mean().item()
        return to_log, score

    def _build_rows(
        self,
        mean_body_dist: Tensor,
        mean_root_dist: Tensor,
    ) -> List[Dict[str, Any]]:
        rows = []

        for motion_id in range(self.motion_lib.num_motions()):
            asset_id, gender, beta_key = self._get_motion_metadata(motion_id)

            matched_env_id = -1
            env_gender = ""
            env_beta_key = ""
            gender_beta_match = False

            if hasattr(self, "_matched_env_id"):
                matched_env_id = int(self._matched_env_id[motion_id].item())

            if matched_env_id >= 0:
                env_gender, env_beta_key = self._get_env_morph_key(matched_env_id)
                gender_beta_match = (gender, beta_key) == (env_gender, env_beta_key)

            rows.append(
                {
                    "motion_id": motion_id,
                    "asset_id": asset_id,
                    "gender": gender,
                    "beta_key": beta_key,
                    "matched_env_id": matched_env_id,
                    "env_gender": env_gender,
                    "env_beta_key": env_beta_key,
                    "gender_beta_match": gender_beta_match,
                    "mean_body_dist": float(mean_body_dist[motion_id].item()),
                    "max_body_dist": float(self._max_body_dist[motion_id].item()),
                    "mean_root_dist": float(mean_root_dist[motion_id].item()),
                    "max_root_dist": float(self._max_root_dist[motion_id].item()),
                    "min_root_height": float(self._min_root_height[motion_id].item()),
                    "steps_seen": int(self._count[motion_id].item()),
                }
            )

        rows.sort(key=lambda x: x["mean_body_dist"], reverse=True)
        return rows

    def _get_motion_metadata(self, motion_id: int) -> Tuple[str, str, str]:
        asset_id = ""
        gender = ""
        beta_key = ""

        if getattr(self.motion_lib, "motion_asset_ids", None) is not None:
            asset_id = str(self.motion_lib.motion_asset_ids[motion_id])
        elif getattr(self.motion_lib, "motion_files", None):
            asset_id = str(self.motion_lib.motion_files[motion_id])

        if getattr(self.motion_lib, "motion_genders", None) is not None:
            gender = str(self.motion_lib.motion_genders[motion_id])
        elif getattr(self.motion_lib, "motion_gender_ids", None) is not None:
            gender_id = int(self.motion_lib.motion_gender_ids[motion_id].item())
            gender = {-1: "female", 0: "neutral", 1: "male"}.get(
                gender_id, str(gender_id)
            )

        if getattr(self.motion_lib, "motion_beta_keys", None) is not None:
            beta_key = str(self.motion_lib.motion_beta_keys[motion_id])

        # Fallback: asset_id follows the existing convention, e.g. male_0e26b88d.
        if (gender == "" or beta_key == "") and asset_id:
            parts = asset_id.split("_", 1)
            if len(parts) == 2:
                gender = gender or parts[0]
                beta_key = beta_key or parts[1]

        return asset_id, gender, beta_key

    def _save_rows(self, rows: List[Dict[str, Any]], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "motion_id",
            "asset_id",
            "gender",
            "beta_key",
            "matched_env_id",
            "env_gender",
            "env_beta_key",
            "gender_beta_match",
            "mean_body_dist",
            "max_body_dist",
            "mean_root_dist",
            "max_root_dist",
            "min_root_height",
            "steps_seen",
        ]

        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nHHI distance report saved to: {output_path}", flush=True)

    def cleanup_after_evaluation(self) -> None:
        self.motion_manager.motion_ids = self._cached_motion_ids
        self.motion_manager.motion_times = self._cached_motion_times
        self.env.restore_state(self._env_snapshot)

        del self._env_snapshot
        del self._cached_motion_ids
        del self._cached_motion_times

        BaseEvaluator.cleanup_after_evaluation(self)
