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
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
from torch import Tensor
import math
from dataclasses import dataclass

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.components.motion_lib import MotionLib
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager


@dataclass
class MimicEpisodeContext:
    """Per-episode-batch state for mimic evaluation."""
    motion_ids: Tensor       # global motion library IDs (used for motion_manager / physics)
    local_ids: Tensor        # 0-based indices into MotionMetrics buffers
    frame_limits: Tensor     # how many frames before clip ends


class MimicEvaluator(BaseEvaluator):
    """Evaluator for Mimic agent's motion tracking performance."""

    def __init__(self, agent: Any, fabric: Any, config: MimicEvaluatorConfig):
        super().__init__(agent, fabric, config)

    @property
    def motion_lib(self) -> MotionLib:
        """Motion library (from agent)."""
        return self.agent.motion_lib

    @property
    def motion_manager(self) -> MimicMotionManager:
        """Motion manager (from env)."""
        return self.env.motion_manager

    def _register_plugins(self) -> None:
        """Register metric computation plugins."""
        self._register_smoothness_plugin(window_sec=0.4, high_jerk_threshold=6500.0)
        self._register_action_smoothness_plugin()

    def _create_metrics(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
        device=None,
    ) -> Dict[str, MotionMetrics]:
        """Create MotionMetrics buffers for trajectory collection (robot state + actions)."""
        if device is None:
            device = self.device
        metrics = {}

        self._add_robot_state_metrics(
            metrics, num_motions, motion_num_frames, max_eval_steps, device=device
        )

        num_dofs = self.env.robot_config.kinematic_info.num_dofs
        metrics["actions"] = MotionMetrics(
            num_motions, motion_num_frames, max_eval_steps, num_dofs, device=device
        )

        return metrics

    def initialize_eval(self) -> Dict:
        """Initialize evaluation tracking and cache env state for restoration."""
        total_motions = self.motion_lib.num_motions()

        # Subsample motions to cap GPU memory usage from MotionMetrics tensors.
        max_eval = self.config.max_eval_motions
        if max_eval is not None and total_motions > max_eval:
            perm = torch.randperm(total_motions, device=self.device)[:max_eval].sort().values
            self._eval_motion_subset = perm
        else:
            self._eval_motion_subset = None

        if self._eval_motion_subset is not None:
            eval_motion_ids = self._eval_motion_subset
            motion_lengths = self.motion_lib.get_motion_length(eval_motion_ids)
        else:
            eval_motion_ids = None
            motion_lengths = self.motion_lib.get_motion_length(None)

        num_motions = motion_lengths.shape[0]
        motion_num_frames = (motion_lengths / self.env.dt).floor().long()
        motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)
        self._init_eval_component_buffers(num_motions)

        # Cache env + motion manager state (restored in cleanup_after_evaluation)
        self._env_snapshot = self.env.save_state()
        self._cached_motion_ids = self.motion_manager.motion_ids.clone()
        self._cached_motion_times = self.motion_manager.motion_times.clone()

        # Store metrics on CPU to avoid exhausting GPU memory for large motion sets.
        return self._create_metrics(
            num_motions, motion_num_frames, self.config.max_eval_steps, device="cpu"
        )

    def _save_failed_motions(self, failed_motions: list, epoch: int) -> None:
        """
        Save list of failed motions to a text file.

        Args:
            failed_motions: List of motion IDs that failed tracking
            epoch: Current epoch number
        """
        filename = f"failed_motions_epoch_{epoch}_rank_{self.fabric.global_rank}.txt"
        self._save_list_to_file(failed_motions, filename, subdirectory="failed_motions")

    def _update_motion_sampling_weights(self) -> None:
        """Update motion sampling weights based on evaluation component failures."""
        if self._motion_failed is None:
            return

        # _motion_failed is indexed by local (eval-subset) positions.
        # Map back to global motion library IDs before updating weights.
        subset = self._eval_motion_subset  # None if evaluating all motions
        local_failed = torch.nonzero(self._motion_failed).flatten()
        local_success = torch.nonzero(~self._motion_failed).flatten()

        if subset is not None:
            global_failed = subset[local_failed].tolist()
            global_success = subset[local_success].tolist()
        else:
            global_failed = local_failed.tolist()
            global_success = local_success.tolist()

        self._save_failed_motions(global_failed, self.agent.current_epoch)

        success_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_success_discount,
            self.config.eval_metrics_every,
        )
        failure_discount = math.pow(
            self.config.motion_weights_rules.motion_weights_update_failure_discount,
            self.config.eval_metrics_every,
        )
        new_weights = self.env.motion_manager.motion_weights.clone()
        new_weights[global_success] *= success_discount
        if failure_discount != 0:
            new_weights[global_failed] /= failure_discount
        else:
            new_weights[global_failed] = 1.0
        self.env.motion_manager.update_sampling_weights(new_weights)

    def evaluate_episode(self, env_ids: torch.Tensor, max_steps: int) -> None:
        """Run a single episode batch, optionally with EMA action smoothing.

        When eval_action_ema_alpha is set, actions are low-pass filtered to
        simulate deployment conditions. Motions that fail under EMA get higher
        sampling weight, creating curriculum pressure toward smooth policies.
        """
        ema_alpha = self.config.eval_action_ema_alpha

        self._on_episode_start(env_ids)

        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        prev_actions = None

        for step_idx in range(max_steps):
            model_outs = self.agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))

            # Apply EMA smoothing (deployment simulation)
            if ema_alpha is not None:
                if prev_actions is None:
                    prev_actions = actions.clone()
                actions = ema_alpha * actions + (1.0 - ema_alpha) * prev_actions
                prev_actions = actions.clone()

            obs, rewards, dones, terminated, extras = self.env.step(actions)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

            self._check_eval_components(env_ids, step_idx)
            self._on_episode_step(env_ids, extras, actions)

    def run_evaluation(self) -> None:
        """Run evaluation across multiple motions."""
        for batch in self._build_eval_batches():
            if len(batch) == 3:
                env_ids, local_ids, actual_ids = batch
            else:
                # fixed-motion path: no subset, local == actual
                env_ids, actual_ids = batch
                local_ids = actual_ids
            motion_lengths = self.motion_lib.get_motion_length(actual_ids)
            max_len = min(
                (motion_lengths.max() / self.env.dt).floor().long().item(),
                self.config.max_eval_steps,
            )
            # Build episode context before evaluate_episode so hooks can read it
            self._episode_ctx = MimicEpisodeContext(
                motion_ids=actual_ids,
                local_ids=local_ids,
                frame_limits=(motion_lengths / self.env.dt).floor().long().clamp(
                    max=self.config.max_eval_steps
                ),
            )
            self.evaluate_episode(env_ids, max_len)

    def _build_eval_batches(self):
        """Build list of (env_ids, motion_ids) batches to evaluate.

        Returns:
            List of (env_ids, motion_ids) tuples where motion_ids are indices
            into self._metrics (0-based within the eval subset).
        """
        fixed_motion_ids, first_env_indices = (
            self.motion_manager.get_unique_fixed_motions()
        )

        if fixed_motion_ids.numel() > 0:
            print(f"Only evaluating fixed motions: {fixed_motion_ids}")
            return [(first_env_indices, fixed_motion_ids)]

        # If we're using a subset, eval_motion_ids maps local index → global motion id.
        if self._eval_motion_subset is not None:
            global_ids = self._eval_motion_subset
            total_eval = global_ids.shape[0]
        else:
            global_ids = None
            total_eval = self.motion_lib.num_motions()

        batches = []
        for start in range(0, total_eval, self.num_envs):
            end = min(start + self.num_envs, total_eval)
            # local_ids: 0-based index into the eval subset (used for MotionMetrics)
            local_ids = torch.arange(start, end, device=self.device)
            # actual_ids: real motion library IDs (used for physics/motion_manager)
            actual_ids = global_ids[local_ids] if global_ids is not None else local_ids
            env_ids = torch.arange(0, local_ids.numel(), device=self.device)
            print(
                f"Evaluating motions {start} to {end} "
                f"(global ids {actual_ids[0].item()}..{actual_ids[-1].item()}), "
                f"out of total eval set {total_eval}"
            )
            batches.append((env_ids, local_ids, actual_ids))
        return batches

    # --- Hook overrides ---
    
    def _on_episode_start(self, env_ids: Tensor) -> None:
        """Set motion_ids/times in the motion manager before reset."""
        self.motion_manager.motion_ids[env_ids] = self._episode_ctx.motion_ids
        self.motion_manager.motion_times[env_ids] = 0.0
    
    def _get_reset_kwargs(self) -> dict:
        """Customize env.reset() for mimic evaluation."""
        return {"sample_flat": True, "disable_motion_resample": True}
    
    def _check_eval_components(self, env_ids: Tensor, step_idx: int) -> None:
        """Filter by frame limits and check failures only for active clips."""
        still_active = self._episode_ctx.frame_limits > step_idx
        if still_active.any():
            active_env_ids = env_ids[still_active]
            # Use local_ids (0-based within eval subset) for the failure buffers,
            # which are sized to num_eval_motions, not total motion library size.
            active_motion_ids = self._episode_ctx.local_ids[still_active]
            self._check_evaluation_failures(active_env_ids, active_motion_ids)

    def _on_episode_step(self, env_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        """Collect smoothness metrics each step."""
        self._record_trajectory_step(
            self._metrics, extras, env_ids, self._episode_ctx.local_ids, actions
        )

    def _record_trajectory_step(
        self,
        metrics: Dict,
        extras: Dict,
        active_env_ids: Tensor,
        active_motion_ids: Tensor,
        actions: Tensor,
    ) -> None:
        """Record robot state and actions into trajectory buffers for this step."""
        if "actions" in metrics and actions is not None:
            metrics["actions"].update(active_motion_ids, actions[active_env_ids].detach())

        for k in metrics.keys():
            if k == "actions":
                continue
            if f"raw/{k}" in extras:
                metrics[k].update(active_motion_ids, extras[f"raw/{k}"][active_env_ids].detach())

    def process_eval_results(self) -> Tuple[Dict, Optional[float]]:
        """Process results and update motion sampling weights."""
        to_log, success_rate = super().process_eval_results()
        self._update_motion_sampling_weights()

        additional_metrics = self._compute_additional_metrics(self._metrics)
        to_log.update(additional_metrics)

        if self.fabric.global_rank == 0:
            if (
                self.config.save_predicted_motion_lib_every is not None
                and self.eval_count % self.config.save_predicted_motion_lib_every == 0
                and self._eval_motion_subset is None  # requires full motion set
            ):
                self._save_predicted_motion_lib(self._metrics, epoch=self.agent.current_epoch)

        return to_log, success_rate

    def cleanup_after_evaluation(self) -> None:
        """Restore env and motion manager state after evaluation."""
        self.motion_manager.motion_ids = self._cached_motion_ids
        self.motion_manager.motion_times = self._cached_motion_times
        self.env.restore_state(self._env_snapshot)

        del self._env_snapshot
        del self._cached_motion_ids
        del self._cached_motion_times
        self._eval_motion_subset = None
        super().cleanup_after_evaluation()

    def _plot_per_frame_metrics(
        self, metrics: Dict, actions_storage: list = None
    ) -> None:
        """
        Plot per-frame metrics vs time when evaluating a single motion.
        Uses base class plotting with custom colors for contact forces.

        Args:
            metrics: Dictionary of MotionMetrics objects
            actions_storage: List of action arrays for plotting (optional, currently unused)
        """
        # Define custom colors for specific metrics
        custom_colors = {}

        # Only plot metrics that were actually collected
        eval_metric_keys = list(self.config.evaluation_components.keys())
        available_keys = [k for k in eval_metric_keys if k in metrics]

        # Use base class generic plotting with custom colors
        super()._plot_per_frame_metrics(
            metrics,
            keys_to_plot=available_keys if available_keys else None,
            custom_colors=custom_colors,
            output_filename="metrics_per_frame_plot.png",
        )

    def _save_predicted_motion_lib(
        self, metrics: Dict[str, MotionMetrics], epoch: int
    ) -> None:
        """Pack collected predicted metrics and save as a MotionLib-compatible .pt file.

        This creates a "predicted" version of MotionLib where unknown fields are copied
        from the ground-truth self.motion_lib.

        Args:
            metrics: Dictionary of MotionMetrics objects containing predicted data
            epoch: Current epoch number for filename
        """
        required_keys = [
            "dof_pos",
            "dof_vel",
            "rigid_body_pos",
            "rigid_body_rot",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "rigid_body_contacts",
        ]

        # Ensure required data exists
        for k in required_keys:
            if k not in metrics:
                raise ValueError(
                    f"Missing metric '{k}' required to build predicted MotionLib"
                )

        device = self.device
        num_motions = self.motion_lib.num_motions()

        motion_num_frames = metrics["dof_pos"].motion_lens.to(device=device).long()
        assert (
            motion_num_frames.shape[0] == num_motions
        ), "motion_num_frames size mismatch"

        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        motion_dt = (
            torch.ones(num_motions, dtype=torch.float32, device=device) * self.env.dt
        )
        motion_lengths = motion_num_frames.to(dtype=torch.float32) * self.env.dt

        def pack_metric(metric_key: str) -> torch.Tensor:
            data = metrics[metric_key].data
            per_motion = []
            for m in range(num_motions):
                f = motion_num_frames[m].item()
                f = min(f, data.shape[1])
                per_motion.append(data[m, :f].detach().clone())
            return torch.cat(per_motion, dim=0)

        # Build packed tensors matching MotionLib field names
        dps = pack_metric("dof_pos")  # [total_frames, num_dofs]
        dvs = pack_metric("dof_vel")  # [total_frames, num_dofs]

        # Rigid body tensors are stored flattened in metrics; reshape to [*, num_bodies, C]
        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        gts_flat = pack_metric("rigid_body_pos")  # [total_frames, num_bodies*3]
        grs_flat = pack_metric("rigid_body_rot")  # [total_frames, num_bodies*4]
        gvs_flat = pack_metric("rigid_body_vel")  # [total_frames, num_bodies*3]
        gavs_flat = pack_metric("rigid_body_ang_vel")  # [total_frames, num_bodies*3]

        # Validate and reshape
        assert (
            gts_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_pos dim mismatch: {gts_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            grs_flat.shape[-1] == num_bodies * 4
        ), f"rigid_body_rot dim mismatch: {grs_flat.shape[-1]} vs {num_bodies*4}"
        assert (
            gvs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_vel dim mismatch: {gvs_flat.shape[-1]} vs {num_bodies*3}"
        assert (
            gavs_flat.shape[-1] == num_bodies * 3
        ), f"rigid_body_ang_vel dim mismatch: {gavs_flat.shape[-1]} vs {num_bodies*3}"

        gts = gts_flat.view(-1, num_bodies, 3)
        grs = grs_flat.view(-1, num_bodies, 4)
        gvs = gvs_flat.view(-1, num_bodies, 3)
        gavs = gavs_flat.view(-1, num_bodies, 3)

        # Pack predicted contacts from metrics
        contacts_data = metrics[
            "rigid_body_contacts"
        ].data  # [num_motions, max_frames, num_bodies]
        contacts_list = []
        for m in range(num_motions):
            f = motion_num_frames[m].item()
            # Clamp to available frames
            f = min(f, contacts_data.shape[1])
            # Convert float contacts to bool for consistency with MotionLib format
            contacts_list.append(contacts_data[m, :f].bool().detach().clone())
        contacts = torch.cat(contacts_list, dim=0)

        # Copy ground-truth motion weights and files
        gt_lib = self.motion_lib
        motion_weights = getattr(
            gt_lib,
            "motion_weights",
            torch.ones(num_motions, dtype=torch.float32, device=device),
        )
        motion_files = getattr(
            gt_lib,
            "motion_files",
            tuple([f"predicted_motion_{i}" for i in range(num_motions)]),
        )

        save_data = {
            "gts": gts,
            "grs": grs,
            "gvs": gvs,
            "gavs": gavs,
            "dvs": dvs,
            "dps": dps,
            "length_starts": length_starts,
            "motion_lengths": motion_lengths,
            "motion_dt": motion_dt,
            "motion_num_frames": motion_num_frames,
            "motion_weights": motion_weights,
            "motion_files": motion_files,
            "contacts": contacts,  # Always save predicted contacts
        }

        # create dir if not exists
        output_dir = self.root_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"predicted_motion_lib_epoch_{epoch}.pt"
        torch.save(save_data, output_path)
        print(f"Predicted MotionLib saved to {output_path}")
