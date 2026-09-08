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
"""Exercise GPU queue orchestration using CPU-only subprocesses and fake data."""

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = "train_150_evalfix_ablation.sh"
EXPERIMENTS = {
    "A": "mlp_wide_discover_historical_lookahead",
    "B": "mlp_wide_discover_attention",
    "C": "mlp_wide_discover_attention_slot_type",
    "D": "mlp_wide_discover_attention_adaln",
    "E": "mlp_wide_discover_attention_slot_type",
}
FAKE_TRAINER = r'''
import json
import os
import sys
import time

args = sys.argv[1:]
name = args[args.index("--experiment-name") + 1]
event = {
    "name": name,
    "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
    "port": os.environ["MASTER_PORT"],
    "pid": os.getpid(),
}
def emit(kind):
    event["kind"] = kind
    with open(os.environ["MOCK_EVENTS"], "a") as f:
        f.write(json.dumps(event) + "\n")

emit("start")
time.sleep(float(os.environ.get("MOCK_DELAY", "0.3")))
emit("end")
sys.exit(7 if name.endswith(os.environ.get("MOCK_FAIL_SUFFIX", "never")) else 0)
'''


@unittest.skipUnless(
    all(shutil.which(command) for command in ("bash", "flock", "setsid")),
    "Requires Linux Bash, flock and setsid",
)
class AblationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="protomotions-ablation-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("tools", "bin", "motions", "examples/experiments/mimic"):
            (self.root / directory).mkdir(parents=True)
        self.script = self.root / "tools" / LAUNCHER
        shutil.copy2(REPO_ROOT / "tools" / LAUNCHER, self.script)
        for experiment in set(EXPERIMENTS.values()):
            (self.root / "examples/experiments/mimic" / (experiment + ".py")).touch()
        for name in ("small150_128shape.pt", "small150_128shape_refined.pt"):
            (self.root / "motions" / name).write_bytes(b"mock; never deserialized")
        trainer = self.root / "bin" / "mock-python"
        trainer.write_text("#!" + sys.executable + "\n" + FAKE_TRAINER)
        trainer.chmod(0o755)
        nvidia = self.root / "bin" / "nvidia-smi"
        nvidia.write_text("#!/bin/sh\nprintf 'mock GPU\\n'\n")
        nvidia.chmod(0o755)
        self.events_file = self.root / "events.jsonl"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("ABLATION_")}
        self.env.update(
            PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
            ABLATION_PYTHON=str(trainer),
            ABLATION_MOTION_DIR=str(self.root / "motions"),
            ABLATION_LOG_DIR=str(self.root / "logs"),
            MOCK_EVENTS=str(self.events_file),
        )

    def run_launcher(self, *flags, extra_env=None):
        env = dict(self.env)
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", str(self.script), *flags],
            env=env, text=True, capture_output=True, timeout=15,
        )

    def events(self):
        if not self.events_file.exists():
            return []
        return [json.loads(line) for line in self.events_file.read_text().splitlines()]

    def test_default_six_commands_match_paper_conditions(self):
        result = self.run_launcher("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 6)
        for gpu, (line, case, seed) in enumerate(
            zip(lines, "ABCDEC", (0, 0, 0, 0, 0, 1))
        ):
            args = shlex.split(line)
            self.assertIn("CUDA_VISIBLE_DEVICES=" + str(gpu), args)
            self.assertIn("MASTER_PORT=" + str(29600 + gpu), args)
            self.assertEqual(args[args.index("--seed") + 1], str(seed))
            self.assertEqual(args[args.index("--ngpu") + 1], "1")
            self.assertEqual(args[args.index("--training-max-steps") + 1], "786432000")
            self.assertIn("agent.evaluator.eval_shape_sampling_seed=42", args)
            self.assertEqual(
                args[args.index("--experiment-path") + 1],
                "examples/experiments/mimic/" + EXPERIMENTS[case] + ".py",
            )
            expected_motion = "small150_128shape" + ("" if case == "E" else "_refined") + ".pt"
            self.assertEqual(Path(args[args.index("--motion-file") + 1]).name, expected_motion)
        self.assertFalse((self.root / "results").exists())
        self.assertFalse((self.root / "logs").exists())

    def test_ten_runs_overlap_across_gpus_but_never_on_one_gpu(self):
        result = self.run_launcher("--all-seeds")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        active, started, peak = set(), set(), 0
        for event in self.events():
            gpu = event["gpu"]
            if event["kind"] == "start":
                self.assertNotIn(gpu, active)
                active.add(gpu)
                started.add(event["name"])
                peak = max(peak, len(active))
            else:
                self.assertIn(gpu, active)
                active.remove(gpu)
            self.assertEqual(int(event["port"]), 29600 + int(gpu))
        self.assertFalse(active)
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 6)
        self.assertEqual(started, {
            "hhi_150_evalfix_" + case + "_seed" + str(seed)
            for case in "ABCDE" for seed in (0, 1)
        })
        self.assertEqual(len(list((self.root / "logs").glob("*.log"))), 10)

    def test_failure_stops_its_queue_but_other_gpus_finish(self):
        result = self.run_launcher("--all-seeds", extra_env={"MOCK_FAIL_SUFFIX": "A_seed0"})
        self.assertNotEqual(result.returncode, 0)
        starts = {event["name"] for event in self.events() if event["kind"] == "start"}
        self.assertNotIn("hhi_150_evalfix_A_seed1", starts)
        self.assertEqual(len(starts), 9)
        self.assertIn("DONE case=E seed=1", result.stdout)

    def test_duplicate_launch_rejected_and_termination_stops_children(self):
        env = dict(self.env, MOCK_DELAY="10")
        process = subprocess.Popen(
            ["bash", str(self.script)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 8
            while len(self.events()) < 6 and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail("Launcher exited before six mock jobs started")
                time.sleep(0.05)
            self.assertEqual(len(self.events()), 6)
            duplicate = self.run_launcher()
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("already active", duplicate.stderr)
        finally:
            process.terminate()
            try:
                process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                # Clean up only this test's recorded mock processes if shutdown regresses.
                for event in self.events():
                    if event["kind"] == "start":
                        try:
                            os.killpg(event["pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                process.kill()
                output, error = process.communicate(timeout=8)
                self.fail("Launcher did not stop its jobs:\n" + output + error)
        self.assertEqual(process.returncode, 130)
        for event in self.events():
            self.assertEqual(event["kind"], "start")
            with self.assertRaises(ProcessLookupError):
                os.kill(event["pid"], 0)


if __name__ == "__main__":
    unittest.main()
