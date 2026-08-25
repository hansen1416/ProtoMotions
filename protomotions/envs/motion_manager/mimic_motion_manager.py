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
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.components.motion_lib import MotionLib
import torch
from typing import Optional


class MimicMotionManager(MotionManager):
    """Motion manager specialized for mimic environments.

    Extends the base MotionManager to handle mimic-specific motion sampling,
    including time progression and conditional resampling on reset.
    """

    config: MimicMotionManagerConfig

    def __init__(
        self,
        config: MimicMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: MotionLib,
        fixed_motion_ids_per_env: Optional[torch.Tensor] = None,
    ):
        """A motion manager that handles motion sampling and tracking for mimic environments.

        Args:
            config: Configuration object containing motion manager settings
            num_envs (int): Number of parallel environments
            env_dt (float): Environment timestep
            device (torch.device): Device to store tensors on
            motion_lib (MotionLib): Motion library containing reference motions
            fixed_motion_ids_per_env (Optional[torch.Tensor], optional): If provided, specifies fixed motion IDs to use for each environment. Defaults to None.
        """
        super().__init__(config, num_envs, env_dt, device, motion_lib, fixed_motion_ids_per_env)

    def get_done_tracks(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Check which motion tracks have reached their end time.

        Args:
            env_ids: Optional tensor of environment indices to check. If None, checks all environments.

        Returns:
            Boolean tensor indicating which motion tracks are done (True) or still playing (False)
        """
        end_times = self.motion_lib.motion_lengths[self.motion_ids]
        done_clip = (self.motion_times + self.env_dt) >= end_times
        if env_ids is not None:
            done_clip = done_clip[env_ids]
        return done_clip

    def post_physics_step(self):
        """Advance motion playback time by one environment timestep.

        Called after each physics simulation step to update the current time
        in each motion track.
        """
        self.motion_times += self.env_dt

    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ):
        """Sample new motions for environments.

        Extends base class with two behaviours:
        1. Mimic-specific: only resample envs whose clip has finished when
           resample_on_reset is False.
        2. Morphology-consistent: when env_asset_ids is set and the motion lib
           contains morphology metadata, restrict sampling to motions that share
           the same gender/betas as each env's humanoid asset.

        Args:
            env_ids: Indices of the environments to reset.
            new_motion_ids: Force specific motion IDs (bypasses morphology sampling).
        """
        # Mimic-specific: only resample envs whose clip has finished
        reset_env_ids = env_ids
        if not self.config.resample_on_reset:
            done_tracks = self.get_done_tracks(env_ids)
            reset_env_ids = env_ids[done_tracks]

        if len(reset_env_ids) == 0:
            return

        # Morphology-consistent sampling: when the caller hasn't forced specific
        # motion IDs, and we know each env's body shape asset, only sample
        # motions that are compatible with that env's gender/betas.
        if (
            new_motion_ids is None
            and self.env_asset_ids is not None
            and self.motion_lib.has_morphology_metadata()
        ):
            asset_ids = [self.env_asset_ids[i.item()] for i in reset_env_ids]
            new_motion_ids = self.motion_lib.sample_motions_for_asset_ids(
                asset_ids,
                source_sampling_weights=self.config.source_sampling_weights,
            )

        # tensor([ 0,  1,  2,  3,  4,  5,  ...], device='cuda:0') 
        # tensor([65, 27,  1,  3, 61, 33, ...], device='cuda:0')
        super().sample_motions(reset_env_ids, new_motion_ids)
