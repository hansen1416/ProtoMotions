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

from types import SimpleNamespace

import torch

from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator
from protomotions.components.motion_lib import MotionLib


def _build_evaluator(clip_to_assets):
    motion_lib = MotionLib.empty(device="cpu")
    motion_lib.motion_clip_ids = tuple(
        clip_id for clip_id, assets in clip_to_assets for _ in assets
    )
    motion_lib.motion_asset_ids = tuple(
        asset_id for _, assets in clip_to_assets for asset_id in assets
    )

    evaluator = object.__new__(MimicEvaluator)
    evaluator.agent = SimpleNamespace(motion_lib=motion_lib)
    return evaluator


def _selected_clip_assets(evaluator, selected):
    motion_lib = evaluator.motion_lib
    return {
        (
            motion_lib.motion_clip_ids[motion_id],
            motion_lib.motion_asset_ids[motion_id],
        )
        for motion_id in selected.tolist()
    }


def test_fixed_shape_panel_is_deterministic_and_has_four_variants_per_clip():
    evaluator = _build_evaluator(
        [
            (f"clip_{clip}", [f"shape_{shape}" for shape in range(8)])
            for clip in range(3)
        ]
    )

    torch.manual_seed(1)
    first = evaluator._sample_fixed_shapes_per_motion(4, seed=42)
    torch.manual_seed(999)
    second = evaluator._sample_fixed_shapes_per_motion(4, seed=42)

    assert torch.equal(first, second)
    selected_clip_ids = [
        evaluator.motion_lib.motion_clip_ids[motion_id]
        for motion_id in first.tolist()
    ]
    assert len(first) == 12
    for clip in range(3):
        assert selected_clip_ids.count(f"clip_{clip}") == 4


def test_fixed_shape_panel_is_stable_when_motion_ids_change():
    shape_ids = [f"shape_{shape}" for shape in range(12)]
    first_evaluator = _build_evaluator(
        [("clip_a", shape_ids), ("clip_b", list(reversed(shape_ids)))]
    )
    second_evaluator = _build_evaluator(
        [("clip_b", shape_ids), ("clip_a", list(reversed(shape_ids)))]
    )

    first = first_evaluator._sample_fixed_shapes_per_motion(4, seed=42)
    second = second_evaluator._sample_fixed_shapes_per_motion(4, seed=42)

    assert _selected_clip_assets(first_evaluator, first) == _selected_clip_assets(
        second_evaluator, second
    )


def test_fixed_shape_panel_uses_all_variants_when_clip_has_fewer_than_requested():
    evaluator = _build_evaluator(
        [
            ("small_clip", ["shape_0", "shape_1", "shape_2"]),
            ("large_clip", [f"shape_{shape}" for shape in range(6)]),
        ]
    )

    selected = evaluator._sample_fixed_shapes_per_motion(4, seed=42)
    selected_clip_ids = [
        evaluator.motion_lib.motion_clip_ids[motion_id]
        for motion_id in selected.tolist()
    ]

    assert len(selected) == 7
    assert selected_clip_ids.count("small_clip") == 3
    assert selected_clip_ids.count("large_clip") == 4
