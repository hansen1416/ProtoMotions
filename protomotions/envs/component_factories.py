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
"""Factory functions for common MdpComponent configurations.

These factories reduce boilerplate in experiment configs by providing
pre-configured MdpComponent instances for frequently used components.

Usage in experiment configs:
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        previous_actions_factory,
        mimic_tracking_rewards_factory,
        tracking_error_term_factory,
    )

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(),
    }

    reward_components = {
        **mimic_tracking_rewards_factory(gt_weight=0.5, gr_weight=0.3),
        "action_smoothness": action_smoothness_factory(weight=-0.02),
    }

MdpComponent Parameters
------------------------

- **compute_func**: Pure tensor function that performs the computation
- **dynamic_vars**: Runtime-resolved context paths (become ONNX inputs)
- **static_params**: Compile-time constants (baked into ONNX graph)

Example:
    MdpComponent(
        compute_func=compute_fn,
        dynamic_vars={"tensor_input": EnvContext.current.dof_pos},  # ONNX input
        static_params={"local_obs": True, "weight": 0.5},           # ONNX constants
    )
"""

import functools
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent


# =============================================================================
# Source Switching (note/README.note.md Section 67)
# =============================================================================

# Per-motion data-source tag values (MotionLib.motion_source_id).
MOTION_SOURCE_CANONICAL = 0  # canonical/AMASS -- shape-invariant DOF-space target
MOTION_SOURCE_HUMOS = 1      # HUMOS -- shape-dependent world-space target, jittery


def _apply_source_mask(compute_func, active_source_id: int, motion_source_id, **kwargs):
    """Module-level body for `_source_masked` -- kept top-level (not a closure) so the
    `functools.partial` wrapping it is picklable (train_agent.py pickles `resolved_configs.pt`,
    including every reward's compute_func, at every launch; a nested closure isn't picklable).
    """
    from protomotions.envs.rewards import mask_reward_by_source

    return mask_reward_by_source(compute_func(**kwargs), motion_source_id, active_source_id)


def _source_masked(compute_func, active_source_id: int):
    """Wrap a reward compute_func so its output is zeroed for envs on another source.

    The wrapped function takes an extra `motion_source_id` kwarg (bound via
    `dynamic_vars` by the caller) alongside whatever `compute_func` already expects. Returns a
    `functools.partial`, not a closure, so it stays picklable.
    """
    return functools.partial(_apply_source_mask, compute_func, active_source_id)


# =============================================================================
# Observation Factories
# =============================================================================


def max_coords_obs_factory(
    use_noisy: bool = False,
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
) -> MdpComponent:
    """Factory for humanoid max-coords observations.

    Args:
        use_noisy: If True, use noisy state (for actor with domain randomization).
        local_obs: If True, use heading-aligned local coordinates.
        root_height_obs: If True, include root height observation.
        observe_contacts: If True, include contact observations.

    Returns:
        MdpComponent configured for max-coords observations.
    """
    from protomotions.envs.obs import compute_humanoid_max_coords_observations

    state = EnvContext.noisy if use_noisy else EnvContext.current
    ground = EnvContext.noisy_ground_heights if use_noisy else EnvContext.ground_heights

    return MdpComponent(
        compute_func=compute_humanoid_max_coords_observations,
        dynamic_vars={
            "body_pos": state.rigid_body_pos,
            "body_rot": state.rigid_body_rot,
            "body_vel": state.rigid_body_vel,
            "body_ang_vel": state.rigid_body_ang_vel,
            "ground_height": ground,
            "body_contacts": EnvContext.body_contacts,
        },
        static_params={
            "local_obs": local_obs,
            "root_height_obs": root_height_obs,
            "observe_contacts": observe_contacts,
            "w_last": True,
        },
    )


def reduced_coords_obs_factory(
    use_noisy: bool = False,
    root_height_obs: bool = False,
    root_vel_obs: bool = False,
) -> MdpComponent:
    """Factory for humanoid reduced-coords observations.

    Args:
        use_noisy: If True, use noisy state (for actor with domain randomization).
        root_height_obs: If True, include root height.
        root_vel_obs: If True, include root linear velocity.

    Returns:
        MdpComponent configured for reduced-coords observations.
    """
    from protomotions.envs.obs import compute_humanoid_reduced_coords_observations

    state = EnvContext.noisy if use_noisy else EnvContext.current
    ground = EnvContext.noisy_ground_heights if use_noisy else EnvContext.ground_heights

    bindings = {
        "dof_pos": state.dof_pos,
        "dof_vel": state.dof_vel,
        "anchor_rot": state.anchor_rot,
        "root_local_ang_vel": state.root_local_ang_vel,
    }

    if root_height_obs:
        bindings["root_pos"] = state.root_pos
        bindings["ground_height"] = ground

    if root_vel_obs:
        bindings["root_rot"] = state.root_rot
        bindings["root_vel"] = state.root_vel

    return MdpComponent(
        compute_func=compute_humanoid_reduced_coords_observations,
        dynamic_vars=bindings,
        static_params={
            "root_height_obs": root_height_obs,
            "root_vel_obs": root_vel_obs,
            "w_last": True,
        },
    )


def historical_max_coords_obs_factory(
    use_noisy: bool = False,
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
    history_steps: Optional[Union[int, list]] = None,
) -> MdpComponent:
    """Factory for historical max-coords observations.

    Args:
        use_noisy: If True, use noisy historical state.
        local_obs: If True, use heading-aligned local coordinates.
        root_height_obs: If True, include root height observation.
        observe_contacts: If True, include contact observations.
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 4, 8, 16]). None = use all.

    Returns:
        MdpComponent configured for historical max-coords observations.
    """
    from protomotions.envs.obs import compute_historical_max_coords_from_state

    hist = EnvContext.noisy_historical if use_noisy else EnvContext.historical

    params = {
        "local_obs": local_obs,
        "root_height_obs": root_height_obs,
        "observe_contacts": observe_contacts,
        "w_last": True,
    }
    if history_steps is not None:
        params["history_steps"] = history_steps

    return MdpComponent(
        compute_func=compute_historical_max_coords_from_state,
        dynamic_vars={
            "historical_rigid_body_pos": hist.rigid_body_pos,
            "historical_rigid_body_rot": hist.rigid_body_rot,
            "historical_rigid_body_vel": hist.rigid_body_vel,
            "historical_rigid_body_ang_vel": hist.rigid_body_ang_vel,
            "historical_ground_heights": hist.ground_heights,
            "historical_body_contacts": hist.body_contacts,
        },
        static_params=params,
    )


def historical_reduced_coords_obs_factory(
    use_noisy: bool = False,
) -> MdpComponent:
    """Factory for historical reduced-coords observations.

    Args:
        use_noisy: If True, use noisy historical state.

    Returns:
        MdpComponent configured for historical reduced-coords observations.
    """
    from protomotions.envs.obs import compute_historical_reduced_coords_from_state

    hist = EnvContext.noisy_historical if use_noisy else EnvContext.historical

    return MdpComponent(
        compute_func=compute_historical_reduced_coords_from_state,
        dynamic_vars={
            "historical_dof_pos": hist.dof_pos,
            "historical_dof_vel": hist.dof_vel,
            "historical_root_rot": hist.root_rot,
            "historical_root_local_ang_vel": hist.root_local_ang_vel,
            "historical_anchor_rot": hist.anchor_rot,
        },
        static_params={"w_last": True},
    )


def previous_actions_factory(
    history_steps: int = 1, processed: bool = False
) -> MdpComponent:
    """Factory for previous actions observation.

    Args:
        history_steps: Number of historical steps to include.
        processed: If True, use processed actions (after tanh/clamp, before PD scaling).
                   If False (default), use raw actions from the policy.

    Returns:
        MdpComponent configured for previous actions.
    """
    from protomotions.envs.obs import compute_historical_actions_from_state

    actions_field = (
        EnvContext.historical.processed_actions
        if processed
        else EnvContext.historical.actions
    )

    return MdpComponent(
        compute_func=compute_historical_actions_from_state,
        dynamic_vars={
            "historical_actions": actions_field,
        },
        static_params={"history_steps": history_steps},
    )


def mimic_target_poses_max_coords_factory(
    use_noisy: bool = False,
    with_velocities: bool = True,
    with_relative: bool = True,
    future_steps: Optional[Union[int, list]] = None,
) -> MdpComponent:
    """Factory for mimic target poses (max-coords format).

    Args:
        use_noisy: If True, use noisy current state for relative computations.
        with_velocities: If True, include velocity information.
        with_relative: If True, include relative pose observations.
        future_steps: Steps to select from MimicControl's future buffer.
            None = use all steps. Int N = first N steps. List = specific step indices.

    Returns:
        MdpComponent configured for max-coords target poses.
    """
    from protomotions.envs.obs import build_max_coords_target_poses

    state = EnvContext.noisy if use_noisy else EnvContext.current

    static_params = {
        "with_velocities": with_velocities,
        "with_relative": with_relative,
        "w_last": True,
    }
    if future_steps is not None:
        static_params["future_steps"] = future_steps

    return MdpComponent(
        compute_func=build_max_coords_target_poses,
        dynamic_vars={
            "current_state_body_pos": state.rigid_body_pos,
            "current_state_body_rot": state.rigid_body_rot,
            "current_state_body_vel": state.rigid_body_vel,
            "current_state_body_ang_vel": state.rigid_body_ang_vel,
            "mimic_ref_pos": EnvContext.mimic.future_pos,
            "mimic_ref_rot": EnvContext.mimic.future_rot,
            "mimic_ref_vel": EnvContext.mimic.future_vel,
            "mimic_ref_ang_vel": EnvContext.mimic.future_ang_vel,
        },
        static_params=static_params,
    )


def mimic_target_poses_future_rel_factory(
    use_noisy: bool = False,
    future_steps: Optional[int] = None,
) -> MdpComponent:
    """Factory for mimic target poses (future-relative format).

    Args:
        use_noisy: If True, use noisy current state for relative computations.
        future_steps: Number of future steps to include. None = use all available.

    Returns:
        MdpComponent configured for future-relative target poses.
    """
    from protomotions.envs.obs import build_max_coords_target_poses_future_rel

    state = EnvContext.noisy if use_noisy else EnvContext.current

    params = {"w_last": True}
    if future_steps is not None:
        params["future_steps"] = future_steps

    return MdpComponent(
        compute_func=build_max_coords_target_poses_future_rel,
        dynamic_vars={
            "current_state_body_pos": state.rigid_body_pos,
            "current_state_body_rot": state.rigid_body_rot,
            "mimic_ref_pos": EnvContext.mimic.future_pos,
            "mimic_ref_rot": EnvContext.mimic.future_rot,
        },
        static_params=params,
    )


def mimic_target_poses_reduced_coords_factory(
    use_noisy: bool = False,
    include_dof_vel: bool = True,
    include_xy_offset: bool = False,
    include_height: bool = False,
    include_anchor_vel: bool = False,
    include_anchor_ang_vel: bool = False,
    zero_xy_offset: bool = False,
) -> MdpComponent:
    """Factory for mimic target poses (reduced-coords format).

    Args:
        use_noisy: If True, use noisy current state.
        include_dof_vel: If True, include DOF velocities.
        include_xy_offset: If True, include XY translation offset in local frame.
        include_height: If True, include absolute height.
        include_anchor_vel: If True, include anchor linear velocity.
        include_anchor_ang_vel: If True, include anchor angular velocity.
        zero_xy_offset: If True, emit zeros for XY offset (for inference).

    Returns:
        MdpComponent configured for reduced-coords target poses.
    """
    from protomotions.envs.obs import build_reduced_coords_target_poses

    state = EnvContext.noisy if use_noisy else EnvContext.current

    return MdpComponent(
        compute_func=build_reduced_coords_target_poses,
        dynamic_vars={
            "current_state_anchor_rot": state.anchor_rot,
            "current_state_anchor_pos": state.anchor_pos,
            "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
            "mimic_ref_anchor_pos": EnvContext.mimic.future_anchor_pos,
            "mimic_ref_dof_vel": EnvContext.mimic.future_dof_vel,
            "mimic_ref_dof_pos": EnvContext.mimic.future_dof_pos,
            "mimic_ref_anchor_vel": EnvContext.mimic.future_anchor_vel,
            "mimic_ref_anchor_ang_vel": EnvContext.mimic.future_anchor_ang_vel,
            "current_ref_anchor_pos": EnvContext.mimic.ref_anchor_pos,
        },
        static_params={
            "include_dof_vel": include_dof_vel,
            "include_xy_offset": include_xy_offset,
            "include_height": include_height,
            "include_anchor_vel": include_anchor_vel,
            "include_anchor_ang_vel": include_anchor_ang_vel,
            "zero_xy_offset": zero_xy_offset,
            "w_last": True,
        },
    )


def mimic_deploy_target_poses_factory(
    use_noisy: bool = False,
    include_dof_vel: bool = True,
    future_steps: Optional[Union[int, List[int]]] = None,
) -> MdpComponent:
    """Factory for deployment-ready mimic target poses.

    Produces observations that only require the robot's anchor orientation (IMU)
    and reference motion data.  No position tracking needed for deployment.

    The observation contains:
    - Reference DOF positions (joint targets, frame-invariant)
    - Reference DOF velocities (optional, frame-invariant)
    - Reference body rotations in current anchor frame (6D per body)

    Args:
        use_noisy: If True, use noisy anchor rotation (for actor with DR).
        include_dof_vel: If True, include DOF velocities.
        future_steps: Steps to select from MimicControl's future buffer.
            None = use all steps.  Int N = first N steps.
            List = specific step indices (1-indexed).

    Returns:
        MdpComponent configured for deploy-ready target poses.
    """
    from protomotions.envs.obs import build_deploy_target_poses

    state = EnvContext.noisy if use_noisy else EnvContext.current

    static_params: Dict[str, Any] = {
        "include_dof_vel": include_dof_vel,
        "w_last": True,
    }
    if future_steps is not None:
        static_params["future_steps"] = future_steps

    return MdpComponent(
        compute_func=build_deploy_target_poses,
        dynamic_vars={
            "current_anchor_rot": state.anchor_rot,
            "mimic_ref_rot": EnvContext.mimic.future_rot,
            "mimic_ref_dof_pos": EnvContext.mimic.future_dof_pos,
            "mimic_ref_dof_vel": EnvContext.mimic.future_dof_vel,
        },
        static_params=static_params,
    )


# =============================================================================
# Reward Factories
# =============================================================================


def action_smoothness_factory(weight: float = -0.02) -> MdpComponent:
    """Factory for action smoothness reward.

    Args:
        weight: Reward weight (typically negative).

    Returns:
        MdpComponent configured for action smoothness.
    """
    from protomotions.envs.rewards import compute_action_smoothness

    return MdpComponent(
        compute_func=compute_action_smoothness,
        dynamic_vars={
            "current_processed_action": EnvContext.current_processed_action,
            "previous_processed_action": EnvContext.previous_processed_action,
        },
        static_params={"weight": weight},
    )


def gt_rew_factory(
    weight: float = 0.5,
    coefficient: float = -100.0,
    worst_k: Optional[int] = None,
    worst_k_alpha: float = 0.0,
    source_mask: Optional[int] = None,
) -> MdpComponent:
    """Factory for position tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.
        worst_k: If set (with worst_k_alpha > 0), blend the whole-body mean error with the
            mean error of the worst_k worst-tracked bodies, so a few badly-tracked bodies
            (e.g. hands during a clap) aren't diluted by many well-tracked ones. No-op by
            default.
        worst_k_alpha: Blend weight for the worst-k term (0.0 = unchanged behavior).
        source_mask: If set (MOTION_SOURCE_CANONICAL or MOTION_SOURCE_HUMOS), zero this
            reward for envs whose current motion isn't that source -- see
            note/README.note.md Section 67. No-op by default.

    Returns:
        MdpComponent configured for position tracking.
    """
    from protomotions.envs.rewards import compute_gt_rew

    compute_func = compute_gt_rew
    dynamic_vars = {
        "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
        "ref_rigid_body_pos": EnvContext.mimic.reward_ref_state.rigid_body_pos,
    }
    if source_mask is not None:
        compute_func = _source_masked(compute_func, source_mask)
        dynamic_vars["motion_source_id"] = EnvContext.mimic.motion_source_id

    return MdpComponent(
        compute_func=compute_func,
        dynamic_vars=dynamic_vars,
        static_params={
            "weight": weight,
            "coefficient": coefficient,
            "worst_k": worst_k,
            "worst_k_alpha": worst_k_alpha,
        },
    )


def gr_rew_factory(
    weight: float = 0.3,
    coefficient: float = -5.0,
    worst_k: Optional[int] = None,
    worst_k_alpha: float = 0.0,
    source_mask: Optional[int] = None,
) -> MdpComponent:
    """Factory for rotation tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.
        worst_k: If set (with worst_k_alpha > 0), blend the whole-body mean error with the
            mean error of the worst_k worst-tracked bodies. No-op by default.
        worst_k_alpha: Blend weight for the worst-k term (0.0 = unchanged behavior).
        source_mask: If set (MOTION_SOURCE_CANONICAL or MOTION_SOURCE_HUMOS), zero this
            reward for envs whose current motion isn't that source -- see
            note/README.note.md Section 67. No-op by default.

    Returns:
        MdpComponent configured for rotation tracking.
    """
    from protomotions.envs.rewards import compute_gr_rew

    compute_func = compute_gr_rew
    dynamic_vars = {
        "current_rigid_body_rot": EnvContext.current.rigid_body_rot,
        "ref_rigid_body_rot": EnvContext.mimic.reward_ref_state.rigid_body_rot,
    }
    if source_mask is not None:
        compute_func = _source_masked(compute_func, source_mask)
        dynamic_vars["motion_source_id"] = EnvContext.mimic.motion_source_id

    return MdpComponent(
        compute_func=compute_func,
        dynamic_vars=dynamic_vars,
        static_params={
            "weight": weight,
            "coefficient": coefficient,
            "worst_k": worst_k,
            "worst_k_alpha": worst_k_alpha,
        },
    )


def gv_rew_factory(weight: float = 0.1, coefficient: float = -0.5) -> MdpComponent:
    """Factory for velocity tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.

    Returns:
        MdpComponent configured for velocity tracking.
    """
    from protomotions.envs.rewards import compute_gv_rew

    return MdpComponent(
        compute_func=compute_gv_rew,
        dynamic_vars={
            "current_rigid_body_vel": EnvContext.current.rigid_body_vel,
            "ref_rigid_body_vel": EnvContext.mimic.reward_ref_state.rigid_body_vel,
        },
        static_params={"weight": weight, "coefficient": coefficient},
    )


def gav_rew_factory(weight: float = 0.1, coefficient: float = -0.1) -> MdpComponent:
    """Factory for angular velocity tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.

    Returns:
        MdpComponent configured for angular velocity tracking.
    """
    from protomotions.envs.rewards import compute_gav_rew

    return MdpComponent(
        compute_func=compute_gav_rew,
        dynamic_vars={
            "current_rigid_body_ang_vel": EnvContext.current.rigid_body_ang_vel,
            "ref_rigid_body_ang_vel": EnvContext.mimic.reward_ref_state.rigid_body_ang_vel,
        },
        static_params={"weight": weight, "coefficient": coefficient},
    )


def rh_rew_factory(
    weight: float = 0.2,
    coefficient: float = -100.0,
    source_mask: Optional[int] = None,
) -> MdpComponent:
    """Factory for root height tracking reward.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.
        source_mask: If set (MOTION_SOURCE_CANONICAL or MOTION_SOURCE_HUMOS), zero this
            reward for envs whose current motion isn't that source -- see
            note/README.note.md Section 67. No-op by default.

    Returns:
        MdpComponent configured for root height tracking.
    """
    from protomotions.envs.rewards import compute_rh_rew

    compute_func = compute_rh_rew
    dynamic_vars = {
        "current_root_height": EnvContext.current.root_height,
        "ref_rigid_body_pos": EnvContext.mimic.reward_ref_state.rigid_body_pos,
    }
    if source_mask is not None:
        compute_func = _source_masked(compute_func, source_mask)
        dynamic_vars["motion_source_id"] = EnvContext.mimic.motion_source_id

    return MdpComponent(
        compute_func=compute_func,
        dynamic_vars=dynamic_vars,
        static_params={"weight": weight, "coefficient": coefficient},
    )


def dp_rew_factory(
    weight: float = 0.5,
    coefficient: float = -40.0,
    source_mask: Optional[int] = None,
) -> MdpComponent:
    """Factory for DOF-angle (joint position) tracking reward.

    Shape-invariant counterpart to gt_rew_factory/gr_rew_factory -- see
    note/README.note.md Section 65.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.
        source_mask: If set (MOTION_SOURCE_CANONICAL or MOTION_SOURCE_HUMOS), zero this
            reward for envs whose current motion isn't that source -- see
            note/README.note.md Section 67. No-op by default.

    Returns:
        MdpComponent configured for DOF-angle tracking.
    """
    from protomotions.envs.rewards import compute_dp_rew

    compute_func = compute_dp_rew
    dynamic_vars = {
        "current_dof_pos": EnvContext.current.dof_pos,
        "ref_dof_pos": EnvContext.mimic.reward_ref_state.dof_pos,
    }
    if source_mask is not None:
        compute_func = _source_masked(compute_func, source_mask)
        dynamic_vars["motion_source_id"] = EnvContext.mimic.motion_source_id

    return MdpComponent(
        compute_func=compute_func,
        dynamic_vars=dynamic_vars,
        static_params={"weight": weight, "coefficient": coefficient},
    )


def dv_rew_factory(
    weight: float = 0.1,
    coefficient: float = -0.5,
    source_mask: Optional[int] = None,
) -> MdpComponent:
    """Factory for DOF-angle velocity tracking reward.

    Shape-invariant counterpart to gv_rew_factory/gav_rew_factory -- see
    note/README.note.md Section 65.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.
        source_mask: If set (MOTION_SOURCE_CANONICAL or MOTION_SOURCE_HUMOS), zero this
            reward for envs whose current motion isn't that source -- see
            note/README.note.md Section 67. No-op by default.

    Returns:
        MdpComponent configured for DOF-angle velocity tracking.
    """
    from protomotions.envs.rewards import compute_dv_rew

    compute_func = compute_dv_rew
    dynamic_vars = {
        "current_dof_vel": EnvContext.current.dof_vel,
        "ref_dof_vel": EnvContext.mimic.reward_ref_state.dof_vel,
    }
    if source_mask is not None:
        compute_func = _source_masked(compute_func, source_mask)
        dynamic_vars["motion_source_id"] = EnvContext.mimic.motion_source_id

    return MdpComponent(
        compute_func=compute_func,
        dynamic_vars=dynamic_vars,
        static_params={"weight": weight, "coefficient": coefficient},
    )


def heading_rew_factory(
    weight: float = 0.1,
    coefficient: float = -10.0,
    source_mask: Optional[int] = None,
) -> MdpComponent:
    """Factory for root heading (yaw-only) tracking reward.

    Near-shape-invariant world-frame grounding -- see note/README.note.md Section 65.

    Args:
        weight: Reward weight.
        coefficient: Exponential coefficient for error.
        source_mask: If set (MOTION_SOURCE_CANONICAL or MOTION_SOURCE_HUMOS), zero this
            reward for envs whose current motion isn't that source -- see
            note/README.note.md Section 67. No-op by default.

    Returns:
        MdpComponent configured for root heading tracking.
    """
    from protomotions.envs.rewards import compute_heading_rew

    compute_func = compute_heading_rew
    dynamic_vars = {
        "current_root_rot": EnvContext.current.root_rot,
        "ref_rigid_body_rot": EnvContext.mimic.reward_ref_state.rigid_body_rot,
    }
    if source_mask is not None:
        compute_func = _source_masked(compute_func, source_mask)
        dynamic_vars["motion_source_id"] = EnvContext.mimic.motion_source_id

    return MdpComponent(
        compute_func=compute_func,
        dynamic_vars=dynamic_vars,
        static_params={"weight": weight, "coefficient": coefficient},
    )


def mimic_dof_tracking_rewards_factory(
    dp_weight: float = 0.5,
    dv_weight: float = 0.1,
    contact_weight: float = -0.1,
    heading_weight: float = 0.1,
    rh_weight: float = 0.2,
    dp_coef: float = -40.0,
    dv_coef: float = -0.5,
    heading_coef: float = -10.0,
    rh_coef: float = -100.0,
) -> Dict[str, MdpComponent]:
    """Factory for shape-invariant (DOF-space) mimic tracking reward bundle.

    Replaces mimic_tracking_rewards_factory's world-space gt/gr/gv/gav terms with DOF-angle
    position/velocity tracking (exactly shape-invariant) plus contact-timing matching
    (already implemented, shape-invariant) and root-heading tracking (near-invariant
    world-frame grounding). See note/README.note.md Section 65.

    Args:
        dp_weight: DOF-angle position tracking weight.
        dv_weight: DOF-angle velocity tracking weight.
        contact_weight: Contact-matching weight (typically negative, it's a penalty).
        heading_weight: Root heading tracking weight.
        rh_weight: Root height tracking weight.
        dp_coef: DOF-angle position coefficient.
        dv_coef: DOF-angle velocity coefficient.
        heading_coef: Root heading coefficient.
        rh_coef: Root height coefficient.

    Returns:
        Dict of MdpComponent instances for DOF-space tracking rewards.
    """
    return {
        "dp_rew": dp_rew_factory(weight=dp_weight, coefficient=dp_coef),
        "dv_rew": dv_rew_factory(weight=dv_weight, coefficient=dv_coef),
        "contact_match_rew": contact_match_rew_factory(weight=contact_weight),
        "heading_rew": heading_rew_factory(weight=heading_weight, coefficient=heading_coef),
        "rh_rew": rh_rew_factory(weight=rh_weight, coefficient=rh_coef),
    }


def mimic_tracking_rewards_factory(
    gt_weight: float = 0.5,
    gr_weight: float = 0.3,
    gv_weight: float = 0.1,
    gav_weight: float = 0.1,
    rh_weight: float = 0.2,
    gt_coef: float = -100.0,
    gr_coef: float = -5.0,
    gv_coef: float = -0.5,
    gav_coef: float = -0.1,
    rh_coef: float = -100.0,
    gt_worst_k: Optional[int] = None,
    gt_worst_k_alpha: float = 0.0,
    gr_worst_k: Optional[int] = None,
    gr_worst_k_alpha: float = 0.0,
) -> Dict[str, MdpComponent]:
    """Factory for standard mimic tracking reward bundle.

    Returns a dict of 5 standard tracking rewards (gt, gr, gv, gav, rh).

    Args:
        gt_weight: Position tracking weight.
        gr_weight: Rotation tracking weight.
        gv_weight: Velocity tracking weight.
        gav_weight: Angular velocity tracking weight.
        rh_weight: Root height tracking weight.
        gt_coef: Position coefficient.
        gr_coef: Rotation coefficient.
        gv_coef: Velocity coefficient.
        gav_coef: Angular velocity coefficient.
        rh_coef: Root height coefficient.
        gt_worst_k: Optional worst-k body count for the position reward (see `gt_rew_factory`).
        gt_worst_k_alpha: Blend weight for gt's worst-k term (0.0 = unchanged behavior).
        gr_worst_k: Optional worst-k body count for the rotation reward (see `gr_rew_factory`).
        gr_worst_k_alpha: Blend weight for gr's worst-k term (0.0 = unchanged behavior).

    Returns:
        Dict of MdpComponent instances for tracking rewards.
    """
    return {
        "gt_rew": gt_rew_factory(
            weight=gt_weight,
            coefficient=gt_coef,
            worst_k=gt_worst_k,
            worst_k_alpha=gt_worst_k_alpha,
        ),
        "gr_rew": gr_rew_factory(
            weight=gr_weight,
            coefficient=gr_coef,
            worst_k=gr_worst_k,
            worst_k_alpha=gr_worst_k_alpha,
        ),
        "gv_rew": gv_rew_factory(weight=gv_weight, coefficient=gv_coef),
        "gav_rew": gav_rew_factory(weight=gav_weight, coefficient=gav_coef),
        "rh_rew": rh_rew_factory(weight=rh_weight, coefficient=rh_coef),
    }


def mimic_source_switched_rewards_factory(
    dp_weight: float = 0.5,
    dv_weight: float = 0.1,
    contact_weight: float = -0.1,
    heading_weight: float = 0.1,
    dp_coef: float = -40.0,
    dv_coef: float = -0.5,
    heading_coef: float = -10.0,
    gt_weight: float = 0.5,
    gr_weight: float = 0.3,
    rh_weight: float = 0.2,
    gt_coef: float = -25.0,
    gr_coef: float = -5.0,
    rh_coef: float = -100.0,
) -> Dict[str, MdpComponent]:
    """Factory for the episode-level source-switched mimic tracking reward bundle.

    Canonical/AMASS episodes (motion_source_id==MOTION_SOURCE_CANONICAL) reward the
    shape-independent DOF-space skill (dp/dv/contact/heading) -- identical to
    `mimic_dof_tracking_rewards_factory`. HUMOS episodes
    (motion_source_id==MOTION_SOURCE_HUMOS) reward the shape-dependent world-space part
    (gt/gr/rh) instead, with coefficients loosened relative to a precision-tracking
    baseline since HUMOS references are known to be jittery. Every term is masked to
    exactly one source, so an episode's reward is driven entirely by whichever source it
    currently draws from. See note/README.note.md Section 67.

    Args:
        dp_weight: DOF-angle position tracking weight (canonical side).
        dv_weight: DOF-angle velocity tracking weight (canonical side).
        contact_weight: Contact-matching weight (canonical side, typically negative).
        heading_weight: Root heading tracking weight (canonical side).
        dp_coef: DOF-angle position coefficient.
        dv_coef: DOF-angle velocity coefficient.
        heading_coef: Root heading coefficient.
        gt_weight: Position tracking weight (HUMOS side).
        gr_weight: Rotation tracking weight (HUMOS side).
        rh_weight: Root height tracking weight (HUMOS side).
        gt_coef: Position coefficient -- loosened vs. a precision-tracking default.
        gr_coef: Rotation coefficient.
        rh_coef: Root height coefficient.

    Returns:
        Dict of MdpComponent instances, each masked to exactly one motion source.
    """
    return {
        # Canonical/AMASS side -- shape-independent DOF-space skill.
        "dp_rew": dp_rew_factory(
            weight=dp_weight, coefficient=dp_coef, source_mask=MOTION_SOURCE_CANONICAL
        ),
        "dv_rew": dv_rew_factory(
            weight=dv_weight, coefficient=dv_coef, source_mask=MOTION_SOURCE_CANONICAL
        ),
        "contact_match_rew": contact_match_rew_factory(
            weight=contact_weight, source_mask=MOTION_SOURCE_CANONICAL
        ),
        "heading_rew": heading_rew_factory(
            weight=heading_weight, coefficient=heading_coef, source_mask=MOTION_SOURCE_CANONICAL
        ),
        # HUMOS side -- shape-dependent world-space adaptation, loosened tolerance.
        "gt_rew": gt_rew_factory(
            weight=gt_weight, coefficient=gt_coef, source_mask=MOTION_SOURCE_HUMOS
        ),
        "gr_rew": gr_rew_factory(
            weight=gr_weight, coefficient=gr_coef, source_mask=MOTION_SOURCE_HUMOS
        ),
        "rh_rew": rh_rew_factory(
            weight=rh_weight, coefficient=rh_coef, source_mask=MOTION_SOURCE_HUMOS
        ),
    }


def pow_rew_factory(
    weight: float = -1e-5,
    min_value: Optional[float] = -0.5,
    use_torque_squared: bool = False,
) -> MdpComponent:
    """Factory for power consumption reward.

    Args:
        weight: Reward weight (typically negative).
        min_value: Optional minimum clamp value.
        use_torque_squared: If True, use torque squared instead of absolute.

    Returns:
        MdpComponent configured for power consumption.
    """
    from protomotions.envs.rewards import compute_pow_rew

    static_params = {"weight": weight, "use_torque_squared": use_torque_squared}
    if min_value is not None:
        static_params["min_value"] = min_value

    return MdpComponent(
        compute_func=compute_pow_rew,
        dynamic_vars={
            "dof_forces": EnvContext.current.dof_forces,
            "dof_vel": EnvContext.current.dof_vel,
        },
        static_params=static_params,
    )


def contact_match_rew_factory(
    weight: float = -0.1,
    zero_during_grace_period: bool = True,
    source_mask: Optional[int] = None,
) -> MdpComponent:
    """Factory for contact matching reward.

    Args:
        weight: Reward weight (typically negative).
        zero_during_grace_period: If True, zero reward during grace period.
        source_mask: If set (MOTION_SOURCE_CANONICAL or MOTION_SOURCE_HUMOS), zero this
            reward for envs whose current motion isn't that source -- see
            note/README.note.md Section 67. No-op by default.

    Returns:
        MdpComponent configured for contact matching.
    """
    from protomotions.envs.rewards import compute_contact_match_rew

    compute_func = compute_contact_match_rew
    dynamic_vars = {
        "sim_contacts": EnvContext.current.rigid_body_contacts,
        "ref_contacts": EnvContext.mimic.ref_state.rigid_body_contacts,
        "contact_body_ids": EnvContext.contact_body_ids,
    }
    if source_mask is not None:
        compute_func = _source_masked(compute_func, source_mask)
        dynamic_vars["motion_source_id"] = EnvContext.mimic.motion_source_id

    return MdpComponent(
        compute_func=compute_func,
        dynamic_vars=dynamic_vars,
        static_params={
            "weight": weight,
            "zero_during_grace_period": zero_during_grace_period,
        },
    )


def contact_force_change_rew_factory(
    weight: float = -1e-5,
    min_value: Optional[float] = -0.5,
    threshold: float = 30.0,
    zero_during_grace_period: bool = True,
) -> MdpComponent:
    """Factory for contact force change reward.

    Args:
        weight: Reward weight (typically negative).
        min_value: Optional minimum clamp value.
        threshold: Force change threshold below which changes are ignored.
        zero_during_grace_period: If True, zero reward during grace period.

    Returns:
        MdpComponent configured for contact force change penalty.
    """
    from protomotions.envs.rewards import compute_contact_force_change_rew

    static_params = {
        "weight": weight,
        "threshold": threshold,
        "zero_during_grace_period": zero_during_grace_period,
    }
    if min_value is not None:
        static_params["min_value"] = min_value

    return MdpComponent(
        compute_func=compute_contact_force_change_rew,
        dynamic_vars={
            "current_contact_force_magnitudes": EnvContext.current_contact_force_magnitudes,
            "prev_contact_force_magnitudes": EnvContext.prev_contact_force_magnitudes,
        },
        static_params=static_params,
    )


# =============================================================================
# Termination Factories
# =============================================================================


def tracking_error_term_factory(threshold: float = 0.5) -> MdpComponent:
    """Factory for tracking error termination.

    Args:
        threshold: Maximum joint error threshold in meters.

    Returns:
        MdpComponent configured for tracking error termination.
    """
    from protomotions.envs.terminations import compute_tracking_error

    return MdpComponent(
        compute_func=compute_tracking_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params={"threshold": threshold},
    )


def mean_tracking_error_term_factory(threshold: float = 0.5) -> MdpComponent:
    """Factory for mean (not max) tracking error termination.

    Unlike tracking_error_term_factory (which resets on any single body's error), this
    terminates only when the AVERAGE error across all bodies exceeds threshold -- tolerates
    one or two bodies drifting from a reference pose that may not be exactly achievable for a
    given body shape, as long as overall tracking stays reasonable. Intended to be combined
    with fall_term_factory (genuine physical failure) rather than used as the only failure
    signal -- see note/README.note.md for the rationale.

    Args:
        threshold: Maximum mean joint error threshold in meters.

    Returns:
        MdpComponent configured for mean tracking error termination.
    """
    from protomotions.envs.terminations import compute_mean_tracking_error

    return MdpComponent(
        compute_func=compute_mean_tracking_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params={"threshold": threshold},
    )


def fall_term_factory(termination_height: float = 0.15) -> MdpComponent:
    """Factory for genuine fall-detection termination (contact + height), independent of
    pose-tracking error.

    An env only terminates here if a body that isn't allowed ground contact (per the robot's
    non_termination_contact_bodies) is BOTH in contact with the ground AND below
    termination_height -- i.e. it has actually fallen over, not merely drifted from the
    reference pose. Meant to be the primary "did it fail" signal, paired with a loosened
    tracking-error termination (mean_tracking_error_term_factory) rather than relying on pose
    deviation alone to end episodes.

    Args:
        termination_height: Height (meters, above local ground) below which a disallowed-
            contact body is considered fallen.

    Returns:
        MdpComponent configured for fall termination.
    """
    from protomotions.envs.terminations import fall_termination

    return MdpComponent(
        compute_func=fall_termination,
        dynamic_vars={
            "rigid_body_pos": EnvContext.current.rigid_body_pos,
            "rigid_body_contacts": EnvContext.current.rigid_body_contacts,
            "ground_heights": EnvContext.ground_heights,
            "non_termination_contact_body_ids": EnvContext.non_termination_contact_body_ids,
            "progress_buf": EnvContext.progress_buf,
        },
        static_params={"termination_height": termination_height},
    )


# =============================================================================
# BeyondMimic Reward Factories
# =============================================================================


def global_anchor_pos_rew_factory(
    weight: float = 0.5, sigma: float = 0.3
) -> MdpComponent:
    """Factory for global anchor position reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.

    Returns:
        MdpComponent configured for global anchor position reward.
    """
    from protomotions.envs.rewards import compute_global_anchor_pos_rew

    return MdpComponent(
        compute_func=compute_global_anchor_pos_rew,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"weight": weight, "sigma": sigma},
    )


def global_anchor_ori_rew_factory(
    weight: float = 0.5, sigma: float = 0.4
) -> MdpComponent:
    """Factory for global anchor orientation reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.

    Returns:
        MdpComponent configured for global anchor orientation reward.
    """
    from protomotions.envs.rewards import compute_global_anchor_ori_rew

    return MdpComponent(
        compute_func=compute_global_anchor_ori_rew,
        dynamic_vars={
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"weight": weight, "sigma": sigma},
    )


def relative_body_pos_rew_factory(
    weight: float = 1.0,
    sigma: float = 0.3,
    use_region_weights: bool = True,
) -> MdpComponent:
    """Factory for relative body position reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.
        use_region_weights: If True, apply region-based body weights.

    Returns:
        MdpComponent configured for relative body position reward.
    """
    from protomotions.envs.rewards import compute_relative_body_pos_rew

    return MdpComponent(
        compute_func=compute_relative_body_pos_rew,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={
            "weight": weight,
            "sigma": sigma,
            "use_region_weights": use_region_weights,
        },
    )


def relative_body_ori_rew_factory(
    weight: float = 1.0,
    sigma: float = 0.4,
    use_region_weights: bool = True,
) -> MdpComponent:
    """Factory for relative body orientation reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.
        use_region_weights: If True, apply region-based body weights.

    Returns:
        MdpComponent configured for relative body orientation reward.
    """
    from protomotions.envs.rewards import compute_relative_body_ori_rew

    return MdpComponent(
        compute_func=compute_relative_body_ori_rew,
        dynamic_vars={
            "current_rigid_body_rot": EnvContext.current.rigid_body_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={
            "weight": weight,
            "sigma": sigma,
            "use_region_weights": use_region_weights,
        },
    )


def global_body_lin_vel_rew_factory(
    weight: float = 1.0,
    sigma: float = 1.0,
    use_region_weights: bool = True,
) -> MdpComponent:
    """Factory for global body linear velocity reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.
        use_region_weights: If True, apply region-based body weights.

    Returns:
        MdpComponent configured for body linear velocity reward.
    """
    from protomotions.envs.rewards import compute_global_body_lin_vel_rew

    return MdpComponent(
        compute_func=compute_global_body_lin_vel_rew,
        dynamic_vars={
            "current_rigid_body_vel": EnvContext.current.rigid_body_vel,
            "ref_rigid_body_vel": EnvContext.mimic.ref_state.rigid_body_vel,
        },
        static_params={
            "weight": weight,
            "sigma": sigma,
            "use_region_weights": use_region_weights,
        },
    )


def global_body_ang_vel_rew_factory(
    weight: float = 1.0,
    sigma: float = 3.14,
    use_region_weights: bool = True,
) -> MdpComponent:
    """Factory for global body angular velocity reward (BeyondMimic).

    Args:
        weight: Reward weight.
        sigma: Gaussian kernel width.
        use_region_weights: If True, apply region-based body weights.

    Returns:
        MdpComponent configured for body angular velocity reward.
    """
    from protomotions.envs.rewards import compute_global_body_ang_vel_rew

    return MdpComponent(
        compute_func=compute_global_body_ang_vel_rew,
        dynamic_vars={
            "current_rigid_body_ang_vel": EnvContext.current.rigid_body_ang_vel,
            "ref_rigid_body_ang_vel": EnvContext.mimic.ref_state.rigid_body_ang_vel,
        },
        static_params={
            "weight": weight,
            "sigma": sigma,
            "use_region_weights": use_region_weights,
        },
    )


# =============================================================================
# BeyondMimic Termination Factories
# =============================================================================


def anchor_pos_error_term_factory(threshold: float = 0.5) -> MdpComponent:
    """Factory for anchor position error termination (BeyondMimic).

    Args:
        threshold: Maximum allowed distance in meters.

    Returns:
        MdpComponent configured for anchor position error termination.
    """
    from protomotions.envs.terminations import compute_anchor_pos_error_term

    return MdpComponent(
        compute_func=compute_anchor_pos_error_term,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"threshold": threshold},
    )


def anchor_ori_error_term_factory(threshold: float = 0.8) -> MdpComponent:
    """Factory for anchor orientation error termination (BeyondMimic).

    Args:
        threshold: Maximum allowed difference in projected gravity z-component.

    Returns:
        MdpComponent configured for anchor orientation error termination.
    """
    from protomotions.envs.terminations import compute_anchor_ori_error_term

    return MdpComponent(
        compute_func=compute_anchor_ori_error_term,
        dynamic_vars={
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"threshold": threshold},
    )


def relative_body_pos_error_term_factory(threshold: float = 0.25) -> MdpComponent:
    """Factory for relative body position error termination (BeyondMimic).

    Args:
        threshold: Maximum allowed error for any body in meters.

    Returns:
        MdpComponent configured for relative body position error termination.
    """
    from protomotions.envs.terminations import compute_relative_body_pos_error_term

    return MdpComponent(
        compute_func=compute_relative_body_pos_error_term,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"threshold": threshold},
    )


def anchor_height_error_term_factory(threshold: float = 0.25) -> MdpComponent:
    """Factory for anchor height error termination.

    Terminates when root height deviates from reference by more than threshold.

    Args:
        threshold: Maximum allowed height deviation in meters.

    Returns:
        MdpComponent configured for anchor height error termination.
    """
    from protomotions.envs.terminations import compute_anchor_height_error_term

    return MdpComponent(
        compute_func=compute_anchor_height_error_term,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params={"threshold": threshold},
    )


# =============================================================================
# Evaluation Metric Factories
# =============================================================================


def gt_error_factory(threshold: float = None) -> MdpComponent:
    """Factory for mean body position error metric.

    Args:
        threshold: If set, fail when mean error > threshold.

    Returns:
        MdpComponent configured for mean body position error evaluation.
    """
    from protomotions.envs.terminations import mean_body_pos_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=mean_body_pos_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params=static_params,
    )


def dp_error_factory(threshold: float = None) -> MdpComponent:
    """Factory for mean DOF-angle (joint position) error metric.

    Shape-invariant counterpart to gt_error_factory -- see note/README.note.md Section 65.
    threshold=0.35 (~20 deg mean per-joint error) is an empirically-calibrated starting
    estimate (FK perturbation sweep against the existing 0.5m gt_error threshold on real
    corpus data), not yet validated against real rollout data.

    Args:
        threshold: If set, fail when mean error > threshold (radians).

    Returns:
        MdpComponent configured for mean DOF-angle error evaluation.
    """
    from protomotions.envs.terminations import mean_dof_pos_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=mean_dof_pos_error,
        dynamic_vars={
            "current_dof_pos": EnvContext.current.dof_pos,
            "ref_dof_pos": EnvContext.mimic.ref_state.dof_pos,
        },
        static_params=static_params,
    )


def max_joint_error_factory(threshold: float = None) -> MdpComponent:
    """Factory for max body position error metric.

    Args:
        threshold: If set, fail when max error > threshold.

    Returns:
        MdpComponent configured for max body position error evaluation.
    """
    from protomotions.envs.terminations import max_body_pos_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=max_body_pos_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params=static_params,
    )


def gr_error_factory(threshold: float = None) -> MdpComponent:
    """Factory for mean body rotation error metric.

    Args:
        threshold: If set, fail when mean error > threshold (radians).

    Returns:
        MdpComponent configured for mean body rotation error evaluation.
    """
    from protomotions.envs.terminations import mean_body_rot_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=mean_body_rot_error,
        dynamic_vars={
            "current_rigid_body_rot": EnvContext.current.rigid_body_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
        },
        static_params=static_params,
    )


def anchor_pos_metric_factory(threshold: float = None) -> MdpComponent:
    """Factory for anchor position error metric.

    Args:
        threshold: If set, fail when error > threshold.

    Returns:
        MdpComponent configured for anchor position error evaluation.
    """
    from protomotions.envs.terminations import anchor_pos_error_value

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=anchor_pos_error_value,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def anchor_ori_metric_factory(threshold: float = None) -> MdpComponent:
    """Factory for anchor orientation error metric.

    Args:
        threshold: If set, fail when error > threshold.

    Returns:
        MdpComponent configured for anchor orientation error evaluation.
    """
    from protomotions.envs.terminations import anchor_ori_error_value

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=anchor_ori_error_value,
        dynamic_vars={
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def relative_body_pos_metric_factory(threshold: float = None) -> MdpComponent:
    """Factory for max relative body position error metric.

    Args:
        threshold: If set, fail when max error > threshold.

    Returns:
        MdpComponent configured for relative body position error evaluation.
    """
    from protomotions.envs.terminations import relative_body_pos_max_error

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=relative_body_pos_max_error,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "current_anchor_rot": EnvContext.current.anchor_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def anchor_height_error_metric_factory(threshold: float = None) -> MdpComponent:
    """Factory for anchor height error metric.

    Args:
        threshold: If set, fail when height error > threshold.

    Returns:
        MdpComponent configured for anchor height error evaluation.
    """
    from protomotions.envs.terminations import anchor_height_error_value

    static_params = {}
    if threshold is not None:
        static_params["threshold"] = threshold
    return MdpComponent(
        compute_func=anchor_height_error_value,
        dynamic_vars={
            "current_anchor_pos": EnvContext.current.anchor_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        },
        static_params=static_params,
    )


def path_distance_error_factory(
    threshold: float = 1.0,
    min_progress: int = 10,
) -> MdpComponent:
    """Factory for path distance evaluation metric.

    Returns a boolean-valued component: True when agent is too far from path.
    Use threshold=0.5 with fail_above=True to convert to failure flag.

    Args:
        threshold: Maximum distance from path (meters).
        min_progress: Minimum steps before checking.

    Returns:
        MdpComponent configured for path distance evaluation.
    """
    from protomotions.envs.terminations import check_path_distance_term

    return MdpComponent(
        compute_func=check_path_distance_term,
        dynamic_vars={
            "head_pos": EnvContext.path.head_pos,
            "target_pos": EnvContext.path.tar_pos,
            "progress_buf": EnvContext.path.progress_buf,
        },
        static_params={
            "fail_dist": threshold,
            "min_progress": min_progress,
            "threshold": 0.5,  # Boolean True (1.0) > 0.5 → fail
        },
    )


def steering_velocity_error_factory(
    speed_tolerance: float = 0.5,
    direction_tolerance: float = 0.7,
) -> MdpComponent:
    """Factory for steering velocity evaluation metric.

    Returns a boolean-valued component: True when velocity deviates too much.
    Use threshold=0.5 with fail_above=True to convert to failure flag.

    Args:
        speed_tolerance: Acceptable speed difference from target (m/s).
        direction_tolerance: Minimum dot product with target direction (0-1).

    Returns:
        MdpComponent configured for steering velocity evaluation.
    """
    from protomotions.envs.terminations import check_steering_velocity_error

    return MdpComponent(
        compute_func=check_steering_velocity_error,
        dynamic_vars={
            "root_pos": EnvContext.current.root_pos,
            "prev_root_pos": EnvContext.steering.prev_root_pos,
            "tar_dir": EnvContext.steering.tar_dir,
            "tar_speed": EnvContext.steering.tar_speed,
            "dt": EnvContext.dt,
        },
        static_params={
            "speed_tolerance": speed_tolerance,
            "direction_tolerance": direction_tolerance,
            "threshold": 0.5,  # Boolean True (1.0) > 0.5 → fail
        },
    )

def morphology_obs_factory() -> MdpComponent:
    """Factory for morphology observation (gender_id + betas).

    Passes through env_morphology from context — shape [num_envs, 11].
    Only valid when using smpl_mor multi-shape assets.

    Returns:
        MdpComponent configured for morphology observations.
    """
    from protomotions.envs.obs.humanoid import compute_morphology_obs

    return MdpComponent(
        compute_func=compute_morphology_obs,
        dynamic_vars={"morphology": EnvContext.env_morphology},
    )


def physics_obs_factory() -> MdpComponent:
    """Factory for physics-feature observation (z-scored segment lengths, mass, widths).

    Passes through env_physics_features from context — shape [num_envs, 15].
    Only valid when using smpl_mor multi-shape assets with physics_features.pt present.

    Returns:
        MdpComponent configured for physics feature observations.
    """
    from protomotions.envs.obs.humanoid import compute_physics_obs

    return MdpComponent(
        compute_func=compute_physics_obs,
        dynamic_vars={"physics_features": EnvContext.env_physics_features},
    )


__all__ = [
    # Observation factories
    "max_coords_obs",
    "reduced_coords_obs",
    "historical_max_coords_obs",
    "historical_reduced_coords_obs",
    "previous_actions",
    "mimic_target_poses_max_coords",
    "mimic_target_poses_future_rel",
    "mimic_target_poses_reduced_coords",
    "mimic_deploy_target_poses",
    "morphology_obs_factory",
    "physics_obs_factory",
    # Reward factories
    "action_smoothness",
    "gt_rew",
    "gr_rew",
    "gv_rew",
    "gav_rew",
    "rh_rew",
    "mimic_tracking_rewards_factory",
    "pow_rew",
    "contact_match_rew",
    "contact_force_change_rew",
    # BeyondMimic reward factories
    "global_anchor_pos_rew",
    "global_anchor_ori_rew",
    "relative_body_pos_rew",
    "relative_body_ori_rew",
    "global_body_lin_vel_rew",
    "global_body_ang_vel_rew",
    # Termination factories
    "tracking_error_term",
    "mean_tracking_error_term",
    "fall_term",
    "anchor_pos_error_term",
    "anchor_ori_error_term",
    "relative_body_pos_error_term",
    "anchor_height_error_term",
    # Evaluation metric factories
    "anchor_height_error_metric_factory",
    "gt_error_factory",
    "max_joint_error_factory",
    "gr_error_factory",
    "anchor_pos_metric_factory",
    "anchor_ori_metric_factory",
    "relative_body_pos_metric_factory",
    "path_distance_error_factory",
    "steering_velocity_error_factory",
    "morphology_obs_factory",
]
