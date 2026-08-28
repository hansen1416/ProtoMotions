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
"""Render motions from a packaged `smpl_mor`/`smpl_mor_neutral` MotionLib .pt file
straight to an MP4 — no trained policy, no checkpoint. Pure kinematic playback:
each frame's pose is read directly from the MotionLib and pushed into the
simulator, exactly like `examples/motion_libs_visualizer_mor.py`'s interactive
viewer, except headless and captured to disk instead of shown on screen.

Intended for pods/remote GPU machines (no local display needed — auto-starts
Xvfb, same pattern as `protomotions/record_video_mor.py`).

One packaged MotionLib file = one env per motion (up to --batch-size motions,
starting at --start). This is exactly the layout `tools/refine_humos_motion.py`'s
before/after comparison files use: build a single interleaved file where even
env indices are the original motion and odd indices are the refined version of
the same clip+shape, then render it here with --layout pairs so each row shows
original (left) next to refined (right).

Examples
--------
    # Single motion file, arrange all motions in a compact grid:
    python tools/render_motionlib_video.py \\
        --motion-file data_cache/refinement_pilot/before_20.pt \\
        --output output/videos/before_20.mp4

    # Before/after comparison, side-by-side pairs (even=before, odd=after):
    python tools/render_motionlib_video.py \\
        --motion-file data_cache/refinement_pilot/before_after_interleaved.pt \\
        --layout pairs \\
        --output output/videos/before_after.mp4

    # Only the first 6 motions, slower spacing:
    python tools/render_motionlib_video.py \\
        --motion-file data_cache/refinement_pilot/before_after_interleaved.pt \\
        --layout pairs --batch-size 6 \\
        --output output/videos/before_after_first3.mp4
"""


def create_parser():
    parser = argparse.ArgumentParser(
        description="Render a packaged MotionLib .pt file to MP4 via kinematic playback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--motion-file", type=str, required=True)
    parser.add_argument(
        "--robot",
        type=str,
        choices=["smpl_mor", "smpl_mor_neutral"],
        default="smpl_mor",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the first motion/env to render.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Number of motions/envs to render starting at --start. 0 = all remaining.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["grid", "pairs"],
        default="grid",
        help=(
            "'grid': roughly-square compact grid, one motion per cell. "
            "'pairs': 2-column layout (col0=even env idx, col1=odd env idx) — "
            "use with before/after interleaved files so each row is one comparison."
        ),
    )
    parser.add_argument(
        "--compact-spawn-spacing",
        type=float,
        default=2.0,
        help="Distance (metres) between adjacent humanoids in the layout.",
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=0.6,
        help="Camera distance as a multiple of env spread — lower zooms in closer. "
        "Try ~0.4 for 4 envs, ~0.6 (default) for 8, higher for larger batches.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frames to record. Defaults to the longest selected motion's length.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video framerate")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MP4 path. Defaults to output/videos/<motion-file-stem>.mp4",
    )
    parser.add_argument(
        "--motion-device",
        type=str,
        default="cpu",
        help="Device for the MotionLib tensors.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Use CPU only for simulation (experimental, GPU is default).",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        default=False,
        help="Don't delete the temp dir of per-frame PNGs after encoding — useful "
        "for debugging a bad video (e.g. black/empty output) by inspecting raw frames.",
    )
    parser.add_argument(
        "--ground",
        action="store_true",
        default=False,
        help=(
            "Spawn a visual checkerboard ground plane (loads a mesh via trimesh — "
            "off by default since it's purely cosmetic and depends on trimesh/asset "
            "setup that can be flaky on some pods; the video works fine without it)."
        ),
    )
    return parser


# --------------------------------------------------------------------------- #
# Module-level: parse args and import simulator before torch                  #
# --------------------------------------------------------------------------- #

import argparse  # noqa: E402
import os  # noqa: E402

parser = create_parser()
args = parser.parse_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("isaacgym")

import logging  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.components.scene_lib import (  # noqa: E402
    MeshSceneObject,
    Scene,
    ObjectOptions,
    SceneLib,
    SceneLibConfig,
    ReplicationMethod,
    SubsetMethod,
)
from protomotions.robot_configs.base import ControlType  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def create_checkerboard_ground(
    num_envs: int, device: torch.device, simulator_type: str = "isaacgym"
) -> SceneLib:
    """Minimal flat checkerboard ground plane, copied verbatim from
    examples/motion_libs_visualizer_mor.py's create_checkerboard_ground()."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    checkerboard_dir = os.path.join(
        project_root, "protomotions/data/assets/checkerboard"
    )

    if simulator_type == "isaaclab":
        asset_path = os.path.join(checkerboard_dir, "checkerboard_ground.usda")
        asset_type = "USD"
    else:
        # IsaacGym, Newton, Genesis use URDF
        asset_path = os.path.join(checkerboard_dir, "checkerboard_ground.urdf")
        asset_type = "URDF"

    if not os.path.exists(asset_path):
        print(f"Warning: Checkerboard ground {asset_type} not found at {asset_path}")
        print(f"Assets should be in: {checkerboard_dir}")
        return None

    # Get texture path for IsaacGym (IsaacLab loads it from USD)
    texture_path = None
    if simulator_type != "isaaclab":
        texture_file = os.path.join(checkerboard_dir, "checkerboard_texture.png")
        if os.path.exists(texture_file):
            texture_path = texture_file

    # Create scenes for each environment
    # IMPORTANT: Each scene needs its own MeshSceneObject instance,
    # otherwise attributes get overwritten during _process_scene_objects()
    scenes = []
    for _ in range(num_envs):
        ground_mesh = MeshSceneObject(
            object_path=asset_path,
            translation=(0.0, 0.0, -0.005),  # Slightly below zero
            rotation=(0.0, 0.0, 0.0, 1.0),  # No rotation (x, y, z, w)
            options=ObjectOptions(
                fix_base_link=True,  # Static object
                vhacd_enabled=False,  # Disable convex decomposition for simple plane
                texture_path=texture_path,  # Texture for IsaacGym (None for IsaacLab)
            ),
        )
        scenes.append(Scene(objects=[ground_mesh], offset=(0.0, 0.0)))

    # Configure scene lib
    scene_lib_config = SceneLibConfig(
        scene_file=None,  # No file, using inline scene
        replicate_method=ReplicationMethod.SEQUENTIAL,
        subset_method=SubsetMethod.FIRST,
        pointcloud_samples_per_object=None,
    )

    # Return a SceneLib without terrain (avoids collision geometry in simulators)
    return SceneLib(
        config=scene_lib_config,
        num_envs=num_envs,
        scenes=scenes,
        device=device,
        terrain=None,  # No terrain to avoid unwanted collisions
    )


def build_layout_offsets(num_envs: int, layout: str, spacing: float) -> torch.Tensor:
    """Return [num_envs, 2] xy offsets for a compact spawn layout (no wide single-line spread)."""
    offsets = torch.zeros(num_envs, 2)

    if layout == "pairs":
        # col = env_id % 2 (0=left/even envs, 1=right/odd envs), row = env_id // 2.
        for env_id in range(num_envs):
            col = env_id % 2
            row = env_id // 2
            offsets[env_id, 0] = col * spacing
            offsets[env_id, 1] = -row * spacing
    else:
        cols = max(1, math.ceil(math.sqrt(num_envs)))
        for env_id in range(num_envs):
            row = env_id // cols
            col = env_id % cols
            offsets[env_id, 0] = col * spacing
            offsets[env_id, 1] = -row * spacing

    return offsets


def setup_fixed_camera(simulator, num_envs: int, distance_scale: float = 0.6):
    """Point the camera at the centroid of all envs and lock it there for recording.

    distance_scale multiplies the env spread to get camera distance — lower is
    closer/more zoomed in. The old hardcoded 1.1 kept ~40 envs' worth of room even
    when only a handful were actually rendered; 0.6 is a much tighter default for
    the small (4-8 env) batches this tool is normally used with.
    """
    import numpy as np
    from isaacgym import gymapi

    gym = simulator._gym
    sim = simulator._sim
    viewer = simulator._viewer

    gym.refresh_actor_root_state_tensor(sim)
    positions = np.array(
        [simulator._get_simulator_root_state(i).root_pos.cpu().numpy() for i in range(num_envs)]
    )
    centroid = positions.mean(axis=0)
    spread = max(
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        4.0,
    )

    cam_pos = gymapi.Vec3(
        float(centroid[0]) - spread * distance_scale,
        float(centroid[1]) - spread * distance_scale,
        float(centroid[2]) + spread * distance_scale * 0.55,
    )
    cam_target = gymapi.Vec3(
        float(centroid[0]),
        float(centroid[1]),
        float(centroid[2]) + 1.0,
    )
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    simulator._update_camera = lambda: gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def main():
    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")
    motion_lib_device = torch.device(args.motion_device)

    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=args.motion_file),
        device=motion_lib_device,
    )

    if not (args.robot in ("smpl_mor", "smpl_mor_neutral") and motion_lib.has_morphology_metadata()):
        raise SystemExit(
            f"{args.motion_file} has no morphology metadata, or --robot={args.robot} is not "
            "a morphology robot. This tool only supports smpl_mor / smpl_mor_neutral files "
            "with per-motion asset_ids (e.g. output of tools/refine_humos_motion.py)."
        )

    total_motions = motion_lib.num_motions()
    start = args.start
    end = (start + args.batch_size) if args.batch_size > 0 else total_motions
    end = min(end, total_motions)
    if start >= total_motions:
        raise SystemExit(
            f"--start {start} out of range: file has {total_motions} motions (0..{total_motions - 1})"
        )

    selected_ids = torch.arange(start, end, device=motion_lib_device, dtype=torch.long)
    num_envs = len(selected_ids)
    env_asset_ids = list(motion_lib.get_motion_asset_ids(selected_ids))
    env_motion_ids = selected_ids.to(device)
    env_motion_lengths = motion_lib.get_motion_num_frames(selected_ids).to(device)

    log.info(f"Rendering {num_envs} motion(s), indices [{start}, {end}) from {args.motion_file}")

    # Lay envs out compactly (grid or before/after pairs) instead of MotionLib's native
    # world positions, so the fixed camera doesn't have to zoom out to fit everything.
    offsets = build_layout_offsets(num_envs, args.layout, args.compact_spawn_spacing).to(
        motion_lib_device
    )
    for i, motion_id in enumerate(selected_ids.tolist()):
        mstart = int(motion_lib.length_starts[motion_id].item())
        mend = mstart + int(motion_lib.motion_num_frames[motion_id].item())
        current_xy = motion_lib.gts[mstart, 0, :2].clone()
        delta_xy = offsets[i] - current_xy
        motion_lib.gts[mstart:mend, :, :2] += delta_xy.view(1, 1, 2)

    robot_cfg = robot_config(args.robot)
    robot_cfg.asset.disable_gravity = True
    robot_cfg.asset.fix_base_link = False
    robot_cfg.asset.self_collisions = False
    robot_cfg.control.control_type = ControlType.TORQUE

    sim_cfg = simulator_config(
        "isaacgym",
        robot_cfg,
        headless=False,
        num_envs=num_envs,
        experiment_name="render_motionlib_video",
    )

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
                "No DISPLAY set and Xvfb is not installed. Run: apt install xvfb"
            )

    if args.ground:
        scene_lib = create_checkerboard_ground(num_envs, device, "isaacgym")
    else:
        # No scene objects — avoids trimesh/mesh-asset loading entirely, which is
        # what create_checkerboard_ground() needs and can be flaky about on a fresh
        # pod. The ground plane is purely cosmetic for kinematic-playback capture.
        scene_lib = SceneLib.empty(num_envs=num_envs, device=device)

    SimulatorClass = get_class(sim_cfg._target_)
    simulator = SimulatorClass(
        config=sim_cfg,
        robot_config=robot_cfg,
        terrain=None,
        device=device,
        scene_lib=scene_lib,
        custom_key_handlers={},
        morphology_asset_ids=env_asset_ids,
    )
    simulator._initialize_with_markers({})

    assert simulator.env_id_to_asset_name == env_asset_ids, (
        "Simulator env->asset assignment does not match the requested motions' asset_ids."
    )

    num_frames = args.num_frames
    if num_frames is None:
        num_frames = int(env_motion_lengths.max().item())
    log.info(f"num_frames={num_frames}")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join("output", "videos", f"{Path(args.motion_file).stem}.mp4")

    frames_tmp = tempfile.mkdtemp(prefix="render_motionlib_video_frames_")
    try:
        gym = simulator._gym
        viewer = simulator._viewer
        if viewer is None:
            raise RuntimeError(
                "simulator._viewer is None (headless=True?) — cannot capture frames."
            )

        zero_actions = torch.zeros(
            num_envs, robot_cfg.kinematic_info.num_dofs, device=device
        )
        env_ids_all = torch.arange(num_envs, device=device)

        def pose_frame(frame_idx: int):
            lib_frame_indices = torch.clamp(
                torch.full_like(env_motion_ids, frame_idx),
                max=env_motion_lengths - 1,
            ).to(motion_lib_device)
            lib_motion_ids = env_motion_ids.to(motion_lib_device)

            state = motion_lib.get_motion_state_exact_frame(lib_motion_ids, lib_frame_indices)

            current_state = simulator.get_robot_state()
            current_state.dof_pos = state.dof_pos.to(device).detach()
            current_state.dof_vel = torch.zeros_like(current_state.dof_pos)
            current_state.rigid_body_pos[:, 0, :] = state.rigid_body_pos.to(device).detach()[:, 0, :]
            current_state.rigid_body_rot[:, 0, :] = state.rigid_body_rot.to(device).detach()[:, 0, :]
            current_state.rigid_body_vel[:, 0, :] = torch.zeros(num_envs, 3, device=device)
            current_state.rigid_body_ang_vel[:, 0, :] = torch.zeros(num_envs, 3, device=device)
            simulator.reset_envs(current_state, env_ids=env_ids_all)

        # Warm-up: pose every env to its real frame-0 position BEFORE locking the
        # camera. Otherwise the camera gets computed from actors' raw/default
        # (pre-reset) root positions — which can be degenerate (all stacked near
        # the origin, or uninitialized GPU memory) — and stays frozen there for
        # every subsequent frame even once reset_envs starts placing actors
        # correctly, producing a video with nothing in frame. Mirrors
        # record_video_mor.py's "env.reset(None) before _setup_fixed_camera" order.
        pose_frame(0)
        simulator.step(zero_actions)
        simulator.render()
        setup_fixed_camera(simulator, num_envs, distance_scale=args.camera_distance_scale)

        log.info(f"Recording {num_frames} frames ...")
        for frame_idx in tqdm(range(num_frames), desc="Recording", unit="frame"):
            pose_frame(frame_idx)
            simulator.step(zero_actions)
            simulator.render()
            gym.write_viewer_image_to_file(viewer, os.path.join(frames_tmp, f"{frame_idx:06d}.png"))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        log.info(f"Encoding → {output_path}")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-r", str(args.fps),
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
        if args.keep_frames:
            log.info(f"Keeping raw frames at {frames_tmp}")
        else:
            shutil.rmtree(frames_tmp, ignore_errors=True)
        if hasattr(simulator, "close"):
            simulator.close()
        if xvfb_proc is not None:
            xvfb_proc.terminate()


if __name__ == "__main__":
    main()
