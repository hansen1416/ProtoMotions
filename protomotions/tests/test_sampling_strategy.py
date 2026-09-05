# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator
from protomotions.components.global_clip_pool import GlobalClipPool, GlobalClipPoolConfig


class _FakeMorphologyMotionLib:
    def __init__(self):
        self.motion_clip_ids = ("clip_a",) * 3 + ("clip_b",) * 3
        self.motion_asset_ids = ("shape_a", "shape_b", "shape_c") * 2
        self._clip_mapping = {
            "clip_a": torch.tensor([0, 1, 2]),
            "clip_b": torch.tensor([3, 4, 5]),
        }

    def has_clip_identity_metadata(self):
        return True

    def has_morphology_metadata(self):
        return True

    def build_clip_id_to_motion_ids(self):
        return self._clip_mapping

    def get_motion_asset_ids(self, motion_ids):
        return tuple(self.motion_asset_ids[index] for index in motion_ids.tolist())


def _make_evaluator(motion_lib, epoch, env_asset_ids=None):
    motion_manager = SimpleNamespace(env_asset_ids=env_asset_ids)
    agent = SimpleNamespace(
        current_epoch=epoch,
        motion_lib=motion_lib,
        env=SimpleNamespace(motion_manager=motion_manager),
        num_envs=len(env_asset_ids) if env_asset_ids is not None else 1,
    )
    evaluator = object.__new__(MimicEvaluator)
    evaluator.agent = agent
    evaluator.fabric = SimpleNamespace(device=torch.device("cpu"))
    evaluator.config = SimpleNamespace(
        eval_metrics_every=10,
        eval_shape_sampling_seed=7,
    )
    evaluator.eval_count = 0
    return evaluator


def test_one_shape_panel_is_deterministic_and_rotates_after_each_eval_epoch():
    motion_lib = _FakeMorphologyMotionLib()
    panel = _make_evaluator(motion_lib, epoch=20)._sample_one_shape_per_motion()
    repeated = _make_evaluator(motion_lib, epoch=20)._sample_one_shape_per_motion()
    next_panel = _make_evaluator(motion_lib, epoch=30)._sample_one_shape_per_motion()

    assert torch.equal(panel, repeated)
    assert panel.numel() == 2
    assert next_panel.numel() == 2

    for base in (0, 3):
        current = panel[(panel >= base) & (panel < base + 3)].item() - base
        following = next_panel[
            (next_panel >= base) & (next_panel < base + 3)
        ].item() - base
        assert following == (current + 1) % 3


def test_morphology_matched_batches_use_every_motion_once_on_matching_envs():
    motion_lib = _FakeMorphologyMotionLib()
    env_asset_ids = ["shape_b", "shape_a", "shape_c"]
    evaluator = _make_evaluator(motion_lib, epoch=20, env_asset_ids=env_asset_ids)
    global_ids = torch.arange(6)

    batches = evaluator._build_morphology_matched_eval_batches(global_ids)

    assert len(batches) == 2
    observed_local_ids = []
    observed_motion_ids = []
    for env_ids, local_ids, motion_ids in batches:
        assert env_ids.unique().numel() == env_ids.numel()
        motion_assets = motion_lib.get_motion_asset_ids(motion_ids)
        for env_id, motion_asset in zip(env_ids.tolist(), motion_assets):
            assert env_asset_ids[env_id] == motion_asset
        observed_local_ids.extend(local_ids.tolist())
        observed_motion_ids.extend(motion_ids.tolist())

    assert sorted(observed_local_ids) == list(range(6))
    assert sorted(observed_motion_ids) == list(range(6))


def test_global_pool_uses_explicit_priority_and_random_slot_quotas():
    pool = object.__new__(GlobalClipPool)
    pool.config = GlobalClipPoolConfig(resident_pool_size=5, random_fraction=0.4)
    pool.global_clip_weights = torch.tensor(
        [0.10, 0.80, 0.30, 0.90, 0.20, 0.70, 0.40, 1.00, 0.60, 0.50]
    )
    pool.global_clip_visit_counts = torch.zeros(10, dtype=torch.long)
    pool.rebuild_count = 0

    torch.manual_seed(13)
    expected_random = torch.randperm(10)[:2]
    remaining_mask = torch.ones(10, dtype=torch.bool)
    remaining_mask[expected_random] = False
    remaining = torch.nonzero(remaining_mask).flatten()
    expected_priority = remaining[
        pool.global_clip_weights[remaining].argsort(descending=True, stable=True)[:3]
    ]

    torch.manual_seed(13)
    selected = pool._select_top_k()

    assert torch.equal(selected[:2], expected_random)
    assert torch.equal(selected[2:], expected_priority)
    assert selected.unique().numel() == 5

