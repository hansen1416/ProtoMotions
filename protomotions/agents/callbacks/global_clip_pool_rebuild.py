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


class GlobalClipPoolRebuildCallback(Callback):
    """Rebuilds a `GlobalClipPool`'s resident clip set on its own epoch cadence.

    Pairs with `GlobalClipPoolConfig` / `GlobalClipPool`
    (protomotions/components/global_clip_pool.py). Only wired in when `agent.motion_lib` is
    actually a `GlobalClipPool` -- see train_agent.py.

    Deliberately triggered on a fixed `pool_rebuild_every` epoch cadence (mirroring
    `MotionShardRotationCallback`'s `before_play_steps` trigger point), NOT on the evaluator's
    `on_eval_end` -- tying resident-set churn to `eval_metrics_every` (default 200 epochs) would
    introduce new/unproven clips *less* often than today's existing shard rotation
    (`epochs_per_shard=64`), a regression rather than an improvement. Weight *updates* still only
    happen when the evaluator runs (`MimicEvaluator._update_motion_sampling_weights`, unaffected
    by this callback); rebuilds just re-rank + swap using whatever weights currently exist, on
    their own faster, independent schedule.
    """

    def on_fit_start(self, agent: PPO) -> None:
        """No-op override.

        Without this, `fabric.call("on_fit_start", agent)` resolves to
        `pytorch_lightning.Callback.on_fit_start(self, trainer, pl_module)` (this class's
        base, unrelated to our single-arg hooks) and crashes with a missing `pl_module`
        argument -- same landmine `AutoResumeCallbackSrun`/`MotionShardRotationCallback` already
        guard against.
        """
        pass

    def on_fit_end(self, agent: PPO) -> None:
        pass

    def before_play_steps(self, agent: PPO) -> None:
        if agent.current_epoch % agent.motion_lib.config.pool_rebuild_every != 0:
            return
        if agent.motion_lib.maybe_rebuild():
            log.info(
                f"[GlobalClipPoolRebuildCallback] rebuilt resident pool "
                f"(rebuild_count={agent.motion_lib.rebuild_count}) at epoch {agent.current_epoch}"
            )
            agent.env.motion_manager.on_motion_lib_reloaded()
            agent._force_full_env_reset = True

    # No `on_load_checkpoint_before_env_load` override needed: unlike `MotionLibPool`'s shard
    # rotation (a pure function of `current_epoch`, so it can be re-synced before the env
    # checkpoint that validates against it is even opened), `GlobalClipPool` residency depends on
    # the persistent scoreboard, which only exists inside that same env checkpoint. The resync
    # (restore scoreboard, recompute top-K, materialize) happens synchronously inside
    # `MotionManager.load_state_dict()` instead, called from `BaseAgent.load()` when the env
    # checkpoint loads. `BaseAgent.fit()` already unconditionally forces a full env reset on the
    # first epoch after any load, so no `_force_full_env_reset` is needed here either.
