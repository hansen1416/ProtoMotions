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
import logging

from pytorch_lightning import Callback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protomotions.agents.ppo import PPO
else:
    PPO = object

log = logging.getLogger(__name__)


class MotionShardRotationCallback(Callback):
    """Rotates a `MotionLibPool`'s loaded shard on epoch boundaries.

    Pairs with `StreamingMotionLibConfig` / `MotionLibPool`
    (protomotions/components/motion_lib_pool.py). Only wired in when
    `agent.motion_lib` is actually a `MotionLibPool` -- see train_agent.py.
    """

    def before_play_steps(self, agent: PPO) -> None:
        if agent.motion_lib.maybe_rotate(agent.current_epoch):
            log.info(
                f"[MotionShardRotationCallback] rotated to shard index "
                f"{agent.motion_lib._current_shard_idx} at epoch {agent.current_epoch}"
            )
            agent._force_full_env_reset = True

    def on_load_checkpoint_end(self, agent: PPO) -> None:
        agent.motion_lib.sync_to_epoch(agent.current_epoch)
        log.info(
            f"[MotionShardRotationCallback] synced motion shard to epoch {agent.current_epoch}"
        )
