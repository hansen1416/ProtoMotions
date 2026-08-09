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
"""Record inference to an MP4 video using IsaacGym's off-screen camera sensor API.

Runs the policy in headless mode (no viewer, no display required).
Frames are piped directly to ffmpeg — no temp PNG files written to disk.

A single motion can be selected from the motion file by index, so you can
record any specific clip without loading the entire dataset into the env.
With --num-envs > 1 and --same-motion, the SAME clip is rendered across up to
--num-envs different body shapes side by side (matched via motion_clip_ids).

Example
-------
    python protomotions/record_video_mor.py \\
        --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \\
        --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \\
        --simulator isaacgym \\
        --motion-index 3 \\
        --video-steps 300 \\
        --output output/videos/g1_motion3.mp4

    # Same clip across 8 different body shapes:
    python protomotions/record_video_mor.py \\
        --checkpoint results/hhi_wide_150motion_128shape_discover/last.ckpt \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
        --simulator isaacgym \\
        --motion-index 3 --num-envs 8 --same-motion \\
        --output output/videos/discover_motion3_8shapes.mp4

    # Same clip across a specific, fixed set of body shapes (e.g. to keep the same
    # 16 shapes consistent across many videos/checkpoints) -- overrides the default
    # "first --num-envs shapes found in file order" behaviour of --same-motion:
    python protomotions/record_video_mor.py \\
        --checkpoint results/hhi_wide_150motion_128shape_discover/last.ckpt \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
        --simulator isaacgym \\
        --motion-index 3 --num-envs 16 --same-motion \\
        --target-asset-ids "male_0e26b88d,female_0e26b88d,male_1a2b3c4d,..." \\
        --output output/videos/discover_motion3_16shapes.mp4
"""


def create_parser():
    parser = argparse.ArgumentParser(
        description="Record policy inference to MP4 via IsaacGym off-screen camera",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use — only 'isaacgym' is supported for video recording",
    )
    parser.add_argument("--motion-file", type=str, default=None)
    parser.add_argument(
        "--motion-index",
        type=int,
        default=0,
        help=(
            "Zero-based index of the motion to record from the .pt motion file. "
            "The selected motion is extracted to a temp file so only that clip is loaded."
        ),
    )
    parser.add_argument(
        "--video-steps",
        type=int,
        default=None,
        help=(
            "Number of simulation steps to record. "
            "Defaults to the length of the selected motion (full clip)."
        ),
    )
    parser.add_argument(
        "--compact-spawn-spacing",
        type=float,
        default=None,
        help="Place humanoids in a compact grid with this spacing (metres).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output MP4 path. Defaults to "
            "output/videos/<experiment_name>-motion<index>-<datetime>.mp4"
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video framerate")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--same-motion",
        action="store_true",
        help=(
            "With --num-envs > 1, render the SAME clip as --motion-index across up to "
            "--num-envs different body shapes (matched via motion_clip_ids), instead of "
            "the default behaviour of an arbitrary motion per shape."
        ),
    )
    parser.add_argument(
        "--target-asset-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated list of asset ids (e.g. 'male_0e26b88d,female_1a2b3c4d') "
            "to use with --same-motion, in the given order, instead of the default "
            "'first --num-envs shapes found in file order'. Use this to keep the exact "
            "same set of body shapes consistent across multiple videos/checkpoints. "
            "A clip missing one of the requested asset ids logs a warning and that slot "
            "is dropped (fewer envs than requested)."
        ),
    )
    parser.add_argument("--scenes-file", type=str, default=None)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value",
    )
    parser.add_argument(
        "--gender-beta",
        nargs="*",
        default=None,
        help="Morphology filter e.g. --gender-beta male:0e26b88d",
    )
    parser.add_argument(
        "--max-motions",
        type=int,
        default=None,
        help="Limit total motions loaded (applied before --motion-index slicing).",
    )
    return parser


# --------------------------------------------------------------------------- #
# Module-level: parse args and import simulator before torch                  #
# --------------------------------------------------------------------------- #

import argparse  # noqa: E402
import os  # noqa: E402

parser = create_parser()
args, _unknown_args = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from datetime import datetime  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def extract_motion_by_index(motion_file: str, index: int):
    """Extract a single motion from a packaged .pt motion file.

    Returns (temp_file_path, asset_id) where asset_id is the morphology asset ID
    for the extracted motion (or None if the file has no motion_asset_ids field).
    The caller is responsible for deleting the temp file afterwards.
    Only works with .pt files; for other formats returns (motion_file, None).
    """
    if not str(motion_file).endswith(".pt"):
        if index != 0:
            log.warning(
                f"--motion-index {index} ignored: only supported for .pt files. "
                "Using the file as-is."
            )
        return motion_file, None

    log.info(f"Loading motion file to extract index {index} ...")
    data = torch.load(motion_file, map_location="cpu", weights_only=False)

    num_motions = len(data["motion_num_frames"])
    if index < 0 or index >= num_motions:
        raise ValueError(
            f"--motion-index {index} is out of range: file has {num_motions} motions "
            f"(valid range 0..{num_motions - 1})"
        )

    ls = data["length_starts"]
    nf = data["motion_num_frames"]
    frame_start = ls[index].item()
    frame_end = (ls[index] + nf[index]).item()

    sliced: dict = {}

    # Per-frame tensors
    for key in ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs"):
        if data.get(key) is not None:
            sliced[key] = data[key][frame_start:frame_end]

    # Per-motion tensors — slice to [index:index+1] and reset length_starts to 0
    sliced["length_starts"] = torch.zeros(1, dtype=ls.dtype)
    for key in (
        "motion_lengths",
        "motion_dt",
        "motion_num_frames",
        "motion_weights",
        "motion_betas",
        "motion_gender_ids",
    ):
        if data.get(key) is not None:
            sliced[key] = data[key][index : index + 1]

    # Per-motion sequences (tuples/lists) — include all present sequence fields
    for key in (
        "motion_genders",
        "motion_beta_keys",
        "motion_asset_ids",
        "motion_files",
        "motion_clip_ids",
        "motion_npz_files",
    ):
        if data.get(key) is not None:
            sliced[key] = data[key][index : index + 1]

    # Extract the asset_id so the caller can pin the simulator to that morphology
    asset_id = None
    if data.get("motion_asset_ids") is not None:
        asset_id = data["motion_asset_ids"][index]

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(sliced, tmp.name)
    tmp.close()
    log.info(
        f"Extracted motion {index} ({frame_end - frame_start} frames)"
        + (f" asset={asset_id}" if asset_id else "")
        + f" → {tmp.name}"
    )
    return tmp.name, asset_id


def _slice_motions_by_indices(data: dict, keep_indices: list) -> dict:
    """Slice a packaged motion dict down to `keep_indices`, rebuilding length_starts."""
    ls = data["length_starts"]
    nf = data["motion_num_frames"]
    frame_keys = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs")
    motion_tensor_keys = (
        "motion_lengths", "motion_dt", "motion_num_frames",
        "motion_weights", "motion_betas", "motion_gender_ids",
    )
    motion_seq_keys = (
        "motion_genders", "motion_beta_keys", "motion_asset_ids",
        "motion_files", "motion_clip_ids", "motion_npz_files",
    )

    # Collect per-frame slices for selected motions.
    frame_slices = [(ls[i].item(), (ls[i] + nf[i]).item()) for i in keep_indices]

    sliced: dict = {}
    for key in frame_keys:
        if data.get(key) is not None:
            sliced[key] = torch.cat([data[key][s:e] for s, e in frame_slices], dim=0)

    # Rebuild length_starts for the compacted frame array.
    new_nf = torch.stack([nf[i] for i in keep_indices])
    sliced["length_starts"] = torch.cat(
        [torch.zeros(1, dtype=ls.dtype), new_nf.cumsum(0)[:-1]]
    )
    sliced["motion_num_frames"] = new_nf

    for key in motion_tensor_keys:
        if key == "motion_num_frames":
            continue
        if data.get(key) is not None:
            sliced[key] = torch.stack([data[key][i] for i in keep_indices])

    for key in motion_seq_keys:
        if data.get(key) is not None:
            sliced[key] = [data[key][i] for i in keep_indices]

    return sliced


def slim_motion_file(motion_file: str, motions_per_shape: int = 1):
    """Reduce a large motion file to K motions per body shape.

    With 8192 motions × 128 shapes this trims from ~3.4 GB down to ~50 MB
    (K=1), making multi-env recording feasible on limited GPU memory.

    Returns the path to a temp .pt file (caller must delete it).
    """
    if not str(motion_file).endswith(".pt"):
        return motion_file

    log.info(f"Slimming motion file to {motions_per_shape} motion(s) per shape ...")
    data = torch.load(motion_file, map_location="cpu", weights_only=False)

    asset_ids = data.get("motion_asset_ids")
    if asset_ids is None:
        log.info("No motion_asset_ids found — using file as-is.")
        return motion_file

    # Pick up to K motion indices per unique asset_id, preserving original order.
    from collections import defaultdict
    counts: dict = defaultdict(int)
    keep_indices = []
    for i, aid in enumerate(asset_ids):
        if counts[aid] < motions_per_shape:
            keep_indices.append(i)
            counts[aid] += 1

    total_in = len(asset_ids)
    total_out = len(keep_indices)
    log.info(
        f"Keeping {total_out}/{total_in} motions "
        f"({len(counts)} shapes × {motions_per_shape})."
    )

    sliced = _slice_motions_by_indices(data, keep_indices)

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(sliced, tmp.name)
    tmp.close()
    log.info(f"Slimmed motion file → {tmp.name}")
    return tmp.name


def extract_motion_across_shapes(
    motion_file: str, index: int, max_shapes: int = None, target_asset_ids: list = None
):
    """Extract body-shape variant(s) of the same clip as `index` into a temp file.

    Uses motion_clip_ids (stable clip identity shared across shapes, see
    MotionLib.build_clip_id_to_motion_ids()) to find all motions that are the same
    underlying clip as the one at `index`, across different body shapes -- so you can
    render e.g. 8 different shapes all performing the same motion side by side.

    If `target_asset_ids` is given, selects exactly those asset ids (by
    motion_asset_ids), in that order, instead of "first max_shapes found in file
    order" -- use this to keep the same set of body shapes consistent across many
    videos/checkpoints rendered from separate invocations of this script. Asset ids
    requested but not present for this clip are skipped with a warning.

    Returns (temp_file_path, num_shapes_found). Caller must delete the temp file.
    """
    if not str(motion_file).endswith(".pt"):
        raise ValueError("Multi-shape recording requires a .pt motion file.")

    log.info(f"Loading motion file to find shape-variants of motion index {index} ...")
    data = torch.load(motion_file, map_location="cpu", weights_only=False)

    clip_ids = data.get("motion_clip_ids")
    if clip_ids is None:
        raise RuntimeError(
            "Motion file has no motion_clip_ids -- cannot match the same clip across "
            "shapes. Use --num-envs 1 or a motion file with clip identity metadata."
        )

    num_motions = len(data["motion_num_frames"])
    if index < 0 or index >= num_motions:
        raise ValueError(
            f"--motion-index {index} is out of range: file has {num_motions} motions "
            f"(valid range 0..{num_motions - 1})"
        )

    target_clip_id = clip_ids[index]
    clip_indices = [i for i, cid in enumerate(clip_ids) if cid == target_clip_id]

    if target_asset_ids:
        asset_ids = data.get("motion_asset_ids")
        if asset_ids is None:
            raise RuntimeError(
                "Motion file has no motion_asset_ids -- cannot select by "
                "--target-asset-ids. Omit --target-asset-ids to use file order."
            )
        asset_id_to_clip_index = {asset_ids[i]: i for i in clip_indices}
        keep_indices = []
        missing = []
        for aid in target_asset_ids:
            if aid in asset_id_to_clip_index:
                keep_indices.append(asset_id_to_clip_index[aid])
            else:
                missing.append(aid)
        if missing:
            log.warning(
                f"Clip {target_clip_id!r} has no motion for {len(missing)} requested "
                f"asset id(s): {missing} -- those slots are dropped."
            )
        log.info(
            f"Motion {index} is clip_id={target_clip_id!r}: matched "
            f"{len(keep_indices)}/{len(target_asset_ids)} requested target asset ids."
        )
    else:
        keep_indices = clip_indices
        if max_shapes is not None:
            keep_indices = keep_indices[:max_shapes]
        log.info(
            f"Motion {index} is clip_id={target_clip_id!r}: found {len(keep_indices)} "
            f"shape-variant(s)" + (f", keeping {max_shapes}" if max_shapes else "")
        )

    sliced = _slice_motions_by_indices(data, keep_indices)

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(sliced, tmp.name)
    tmp.close()
    log.info(f"Extracted {len(keep_indices)} shape-variant(s) → {tmp.name}")
    return tmp.name, len(keep_indices)


def parse_gender_beta_filter(gender_beta_args):
    if not gender_beta_args:
        return None
    selected_asset_ids = []
    valid_genders = {"male", "female", "neutral"}
    for item in gender_beta_args:
        if ":" not in item:
            raise ValueError(
                f"Invalid --gender-beta item '{item}'. Expected format: gender:beta_key"
            )
        gender, beta_key = item.split(":", 1)
        gender = gender.strip().lower()
        beta_key = beta_key.strip()
        if gender not in valid_genders:
            raise ValueError(f"Invalid gender '{gender}'.")
        if not beta_key:
            raise ValueError(f"Empty beta_key in '{item}'.")
        selected_asset_ids.append(f"{gender}_{beta_key}")
    return list(dict.fromkeys(selected_asset_ids))


def _setup_fixed_camera(simulator, env):
    """Point the camera at the centroid of all envs and lock it there for recording."""
    import numpy as np
    from isaacgym import gymapi

    gym = simulator._gym
    sim = simulator._sim
    viewer = simulator._viewer

    gym.refresh_actor_root_state_tensor(sim)
    positions = np.array(
        [simulator._get_simulator_root_state(i).root_pos.cpu().numpy()
         for i in range(env.num_envs)]
    )
    centroid = positions.mean(axis=0)
    spread = max(
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        4.0,
    )

    # Diagonal view — camera behind-and-left, looking at centroid
    cam_pos = gymapi.Vec3(
        float(centroid[0]) - spread * 1.5,
        float(centroid[1]) - spread * 1.5,
        float(centroid[2]) + spread * 0.5,
    )
    cam_target = gymapi.Vec3(
        float(centroid[0]),
        float(centroid[1]),
        float(centroid[2]) + 1.0,
    )
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Replace the per-frame tracking update with a no-op so the view stays fixed.
    simulator._update_camera = lambda: gym.viewer_camera_look_at(
        viewer, None, cam_pos, cam_target
    )


def record_video(agent, env, simulator, num_steps, fps, output_path):
    """Record inference using the IsaacGym viewer and encode directly to MP4."""
    import shutil
    from tqdm import tqdm

    gym = simulator._gym
    viewer = simulator._viewer
    if viewer is None:
        raise RuntimeError(
            "simulator._viewer is None — simulator was created with headless=True. "
            "Ensure headless=False in main() and that DISPLAY is set."
        )

    agent.eval()

    # Warm-up: one reset+render so root states are populated, then fix the camera.
    obs, _ = env.reset(None)
    simulator.render()
    _setup_fixed_camera(simulator, env)

    frames_tmp = tempfile.mkdtemp(prefix="record_video_frames_")
    try:
        log.info(f"Recording {num_steps} frames ...")
        done_indices = None
        for step in tqdm(range(num_steps), desc="Recording", unit="frame"):
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            model_outs = agent.model(obs_td)
            action = model_outs.get("mean_action", model_outs["action"])
            obs, rewards, dones, terminated, extras = env.step(action)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

            simulator.render()
            gym.write_viewer_image_to_file(
                viewer, os.path.join(frames_tmp, f"{step:06d}.png")
            )
            obs, _ = env.reset(done_indices)

        if num_steps < 2:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        log.info(f"Encoding → {output_path}")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-r", str(fps),
                "-i", os.path.join(frames_tmp, "%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                "-movflags", "+faststart",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"Saved → {output_path}")
    finally:
        shutil.rmtree(frames_tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main():
    global parser, args
    args = parser.parse_args()

    if args.simulator != "isaacgym":
        raise SystemExit(
            f"record_video_mor.py only supports --simulator isaacgym "
            f"(got '{args.simulator}'). Other simulators lack the camera sensor API."
        )

    checkpoint = Path(args.checkpoint)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_configs_path.exists(), (
        f"Could not find resolved configs at {resolved_configs_path}"
    )

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

    # Use the viewer (same render path as inference_agent_mor.py).
    # Requires a DISPLAY — auto-start Xvfb if none is set.
    xvfb_proc = None
    if not os.environ.get("DISPLAY"):
        import time
        xvfb_display = ":99"
        try:
            xvfb_proc = subprocess.Popen(
                ["Xvfb", xvfb_display, "-screen", "0", "1920x1080x24"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)
            os.environ["DISPLAY"] = xvfb_display
            log.info(f"Started Xvfb on {xvfb_display}")
        except FileNotFoundError:
            raise RuntimeError(
                "No DISPLAY env var set and Xvfb is not installed. "
                "Run: apt install xvfb  OR  set DISPLAY before calling this script."
            )
    simulator_config.headless = False
    log.info(f"Using viewer-based recording (DISPLAY={os.environ.get('DISPLAY')})")

    # Switch simulator if needed (e.g. checkpoint was trained with different sim)
    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # CLI overrides
    if args.num_envs is not None:
        simulator_config.num_envs = args.num_envs

    motion_file = args.motion_file or motion_lib_config.motion_file
    if args.max_motions is not None:
        motion_lib_config.max_motions = args.max_motions

    # Extract a single motion by index into a temp file.
    # Also returns the asset_id of that motion so we can pin the simulator to
    # exactly that one morphology — otherwise round-robin over 128 assets would
    # assign env 0 a shape that has no motion in the single-motion file.
    tmp_motion_file = None
    extracted_asset_id = None
    if motion_file is not None:
        if args.num_envs == 1:
            tmp_motion_file, extracted_asset_id = extract_motion_by_index(
                motion_file, args.motion_index
            )
            motion_lib_config.motion_file = tmp_motion_file
        elif args.same_motion:
            # Multi-env, same clip: render --motion-index's clip across up to num_envs
            # different body shapes, matched via motion_clip_ids (or via the exact,
            # ordered --target-asset-ids list if one was given).
            target_asset_ids = None
            if args.target_asset_ids:
                target_asset_ids = [
                    a.strip() for a in args.target_asset_ids.split(",") if a.strip()
                ]
            tmp_motion_file, num_found = extract_motion_across_shapes(
                motion_file,
                args.motion_index,
                max_shapes=args.num_envs,
                target_asset_ids=target_asset_ids,
            )
            motion_lib_config.motion_file = tmp_motion_file
            expected = len(target_asset_ids) if target_asset_ids else args.num_envs
            if num_found < expected:
                log.warning(
                    f"Only {num_found} shape-variant(s) found for motion "
                    f"{args.motion_index} (requested {expected}); "
                    "some envs will round-robin repeated shapes."
                )
        else:
            # Multi-env: slim to 1 motion per shape so we don't load the full 3+ GB file.
            # Round-robin assigns each env a different body shape from the slimmed set.
            tmp_motion_file = slim_motion_file(motion_file, motions_per_shape=1)
            motion_lib_config.motion_file = tmp_motion_file
            log.info(
                f"num_envs={args.num_envs} > 1: slimmed motion file loaded "
                f"(--motion-index {args.motion_index} ignored for multi-env recording; "
                "pass --same-motion to render this clip across multiple shapes instead)."
            )

    # Auto-detect video length from the motion file if not explicitly set.
    video_steps = args.video_steps
    if video_steps is None:
        active_file = tmp_motion_file or motion_file
        if active_file and str(active_file).endswith(".pt"):
            meta = torch.load(active_file, map_location="cpu", weights_only=False)
            nf = meta["motion_num_frames"]
            if args.num_envs == 1:
                video_steps = int(nf[0].item())
            else:
                video_steps = int(nf.max().item())
            log.info(f"Auto video-steps = {video_steps} (from motion file)")
        else:
            video_steps = 300
            log.info(f"Could not read motion length; defaulting to {video_steps} steps.")

    if args.scenes_file is not None:
        scene_lib_config.scene_file = args.scenes_file

    if args.compact_spawn_spacing is not None:
        log.info(f"CLI override: compact_spawn_spacing = {args.compact_spawn_spacing}")
        env_config.compact_spawn_spacing = args.compact_spawn_spacing

    # Morphology pinning: --gender-beta takes priority; otherwise use the
    # asset_id that came from the extracted motion.
    selected_asset_ids = parse_gender_beta_filter(args.gender_beta)
    if selected_asset_ids is not None:
        robot_config.asset.selected_asset_ids = selected_asset_ids
    elif extracted_asset_id is not None and args.num_envs == 1:
        robot_config.asset.selected_asset_ids = [extracted_asset_id]
        log.info(f"Auto-pinned morphology to asset: {extracted_asset_id}")
    elif extracted_asset_id is not None:
        log.info(
            f"num_envs={args.num_envs} > 1: skipping morphology pin ({extracted_asset_id}), "
            "using round-robin across available body shapes."
        )

    from protomotions.utils.config_utils import parse_cli_overrides, apply_config_overrides

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

    fabric_config = FabricConfig(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None) if hasattr(env_config, "save_dir") else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=checkpoint.parent,
    )
    agent.setup()
    agent.load(str(args.checkpoint), load_env=False)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        experiment_name = getattr(simulator_config, "experiment_name", "inference")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_path = os.path.join(
            "output", "videos",
            f"{experiment_name}-motion{args.motion_index}-{timestamp}.mp4",
        )

    try:
        record_video(
            agent=agent,
            env=env,
            simulator=simulator,
            num_steps=video_steps,
            fps=args.fps,
            output_path=output_path,
        )
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()
        if xvfb_proc is not None:
            xvfb_proc.terminate()
        if tmp_motion_file and tmp_motion_file != args.motion_file:
            try:
                os.unlink(tmp_motion_file)
                log.info(f"Removed temp motion file {tmp_motion_file}")
            except OSError:
                pass


if __name__ == "__main__":
    main()
