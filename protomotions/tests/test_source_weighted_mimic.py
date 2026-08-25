# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator
from protomotions.components.global_clip_pool import GlobalClipPool
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.component_factories import (
    build_source_weighted_reward_table,
    dp_error_factory,
    gt_error_factory,
    mimic_source_weighted_rewards_factory,
)
from tools.build_mixed_source_corpus import (
    MOTION_SOURCE_CANONICAL,
    MOTION_SOURCE_HUMOS,
    _tag_source,
    _validate_aligned_sources,
)


def _aligned_source_dict():
    return {
        "motion_num_frames": torch.tensor([10, 20]),
        "motion_clip_ids": ("clip_a", "clip_b"),
        "motion_asset_ids": ("shape_a", "shape_b"),
        "motion_beta_keys": ("beta_a", "beta_b"),
        "motion_genders": ("neutral", "neutral"),
        "motion_gender_ids": torch.tensor([0, 0]),
        "motion_betas": torch.zeros(2, 16),
    }


def _source_motion_lib():
    motion_lib = MotionLib.empty(device="cpu")
    motion_lib.motion_lengths = torch.ones(4)
    motion_lib.motion_weights = torch.tensor([100.0, 1.0, 1.0, 100.0])
    motion_lib.motion_asset_ids = ("shape", "shape", "shape", "shape")
    motion_lib.motion_source_id = torch.tensor([0, 0, 1, 1])
    motion_lib.motion_clip_ids = (
        "amass::clip",
        "amass::clip",
        "humos::clip",
        "humos::clip",
    )
    return motion_lib


def test_mixed_corpus_uses_source_aware_task_ids_and_preserves_base_ids():
    canonical = _tag_source(
        _aligned_source_dict(), MOTION_SOURCE_CANONICAL, "canonical.pt"
    )
    humos = _tag_source(_aligned_source_dict(), MOTION_SOURCE_HUMOS, "humos.pt")

    _validate_aligned_sources(canonical, humos)
    combined = GlobalClipPool._concat_clip_dicts([canonical, humos])

    assert combined["motion_base_clip_ids"] == (
        "clip_a", "clip_b", "clip_a", "clip_b"
    )
    assert combined["motion_clip_ids"] == (
        "amass::clip_a", "amass::clip_b", "humos::clip_a", "humos::clip_b"
    )
    assert torch.equal(combined["motion_source_id"], torch.tensor([0, 0, 1, 1]))


def test_mixed_corpus_rejects_misaligned_rows():
    canonical = _tag_source(
        _aligned_source_dict(), MOTION_SOURCE_CANONICAL, "canonical.pt"
    )
    humos_data = _aligned_source_dict()
    humos_data["motion_asset_ids"] = ("shape_a", "wrong_shape")
    humos = _tag_source(humos_data, MOTION_SOURCE_HUMOS, "humos.pt")

    try:
        _validate_aligned_sources(canonical, humos)
    except RuntimeError as exc:
        assert "motion_asset_ids" in str(exc)
    else:
        raise AssertionError("Expected misaligned source rows to be rejected")


def test_source_weighted_reward_profiles_are_balanced_and_applied_per_episode():
    canonical = build_source_weighted_reward_table(0.75)
    humos = build_source_weighted_reward_table(0.30)

    positive_terms = ("dp", "dv", "heading", "gt", "gr", "gv", "gav", "rh")
    assert abs(sum(canonical[key] for key in positive_terms) - 1.3) < 1e-7
    assert abs(sum(humos[key] for key in positive_terms) - 1.3) < 1e-7
    assert canonical["dp"] > humos["dp"]
    assert humos["gt"] > canonical["gt"]
    assert all(canonical[key] > 0 and humos[key] > 0 for key in positive_terms)

    components = mimic_source_weighted_rewards_factory()
    assert len(components) == 9
    dp_component = components["dp_rew"]
    assert dp_component.static_params["weight"] == 1.0
    assert "motion_source_id" in dp_component.dynamic_vars

    reward = dp_component.compute_func(
        current_dof_pos=torch.zeros(2, 3),
        ref_dof_pos=torch.zeros(2, 3),
        motion_source_id=torch.tensor([0, 1]),
        coefficient=dp_component.static_params["coefficient"],
    )
    assert torch.allclose(
        reward, torch.tensor([canonical["dp"], humos["dp"]])
    )


def test_source_first_sampling_respects_ratio_before_curriculum_weights():
    motion_lib = _source_motion_lib()

    assert motion_lib.sample_motions_for_asset_ids(
        ["shape"], deterministic=True, source_sampling_weights=[1.0, 0.0]
    ).item() == 0
    assert motion_lib.sample_motions_for_asset_ids(
        ["shape"], deterministic=True, source_sampling_weights=[0.0, 1.0]
    ).item() == 2

    torch.manual_seed(7)
    sampled = motion_lib.sample_motions_for_asset_ids(
        ["shape"] * 1000, source_sampling_weights=[0.5, 0.5]
    )
    humos_fraction = motion_lib.motion_source_id[sampled].float().mean().item()
    assert 0.45 < humos_fraction < 0.55


def test_evaluator_separates_sources_and_uses_amass_dp_for_selection():
    motion_lib = _source_motion_lib()
    config = MimicEvaluatorConfig(
        evaluation_components={
            "dp_error": dp_error_factory(threshold=0.35),
            "gt_error": gt_error_factory(threshold=0.5),
        },
        save_predicted_motion_lib_every=None,
        source_success_components={0: ["dp_error"], 1: ["gt_error"]},
        score_metric="eval_amass/dp_error/mean",
        score_greater_is_better=False,
    )
    evaluator = object.__new__(MimicEvaluator)
    evaluator.agent = SimpleNamespace(motion_lib=motion_lib)
    evaluator.fabric = SimpleNamespace(device=torch.device("cpu"), global_rank=1)
    evaluator.config = config
    evaluator._motion_failed = torch.tensor([False, True, True, True])
    evaluator._per_component_failures = {
        "dp_error": torch.tensor([False, True, True, True]),
        "gt_error": torch.tensor([True, True, False, True]),
    }
    evaluator._component_value_sum = {
        "dp_error": torch.tensor([1.0, 3.0, 10.0, 14.0]),
        "gt_error": torch.tensor([8.0, 10.0, 2.0, 4.0]),
    }
    evaluator._component_step_count = {
        "dp_error": torch.tensor([2, 2, 2, 2]),
        "gt_error": torch.tensor([2, 2, 2, 2]),
    }
    evaluator._component_value_max = {
        "dp_error": torch.tensor([0.6, 1.6, 5.5, 7.5]),
        "gt_error": torch.tensor([4.5, 5.5, 1.5, 2.5]),
    }
    evaluator._component_value_min = {
        "dp_error": torch.tensor([0.4, 1.4, 4.5, 6.5]),
        "gt_error": torch.tensor([3.5, 4.5, 0.5, 1.5]),
    }
    evaluator._eval_motion_subset = None
    evaluator._metrics = {}
    evaluator._update_motion_sampling_weights = lambda: None
    evaluator._compute_additional_metrics = lambda metrics: {}

    to_log, score = evaluator.process_eval_results()

    assert to_log["eval_amass/success_rate"] == 0.5
    assert to_log["eval_humos/success_rate"] == 0.5
    assert to_log["eval/source_conditioned_success_rate"] == 0.5
    assert to_log["eval_amass/dp_error/mean"] == 1.0
    assert to_log["eval_humos/dp_error/mean"] == 6.0
    assert score == -1.0
    assert to_log["eval/selection_score"] == -1.0

    expanded = evaluator._expand_to_clip_variants(torch.tensor([0]))
    assert set(expanded.tolist()) == {0, 1}
