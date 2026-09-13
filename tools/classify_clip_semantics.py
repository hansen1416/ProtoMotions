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
"""
Semantic classifier for the 150 clips in `small150_128shape_refined.pt`, over the class
vocabulary the paper (`paper/sections/08_physical_analysis.tex`) already commits to:
walking, running, jumping, kicking, dancing, sitting -- plus an explicit `other`
bucket for anything that doesn't keyword-match. See note/README.physical-analysis-plan.md
section 1.

Simple keyword-rule classifier, not an LLM/embedding classifier -- 150 items is small
enough that the actual validation step is a human reading the full assignment table
(this script's output), not classifier sophistication.

A clip's 4 captions are joined into one string and matched against every class (assign-
to-all, not first-match-only) -- confirmed against the real multi-match rate below, per
the plan's open decision #1.

Usage:
    python tools/classify_clip_semantics.py \\
        --annotations data_cache/annotations_processed_150.json \\
        --clip-ids data_cache/small150_128shape.clip_ids.txt \\
        --output data_cache/clip_semantic_classes.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Order doesn't matter for assign-to-all matching, but keep it stable for the audit table.
CLASS_KEYWORDS: Dict[str, str] = {
    "walking": r"\b(walk\w*|stroll\w*|march\w*|saunter\w*|amble\w*|stride\w*)\b",
    "running": r"\b(run\w*|jog\w*|sprint\w*|dash\w*)\b",
    "jumping": r"\b(jump\w*|hop\w*|hops|leap\w*|bound\w*)\b",
    "kicking": r"\b(kick\w*)\b",
    "dancing": r"\b(danc\w*)\b",
    "sitting": r"\b(sit\w*|sat|seated)\b",
    # The vocabulary above is the paper's own classification example list, verbatim
    # (paper/sections/08_physical_analysis.tex). "standing" isn't in that list, but the
    # paper's hypothesis paragraph explicitly names it as one of the two quasi-static
    # comparator classes ("standing, sitting") -- and without it, ~1/3 of the true
    # standing-still clips were falling into the generic `other` bucket (see
    # note/README.physical-analysis-plan.md follow-up), inflating its heterogeneity.
    "standing": r"\b(stand\w*|motionless|idle)\b",
}


def classify_clip(caption_text: str) -> List[str]:
    matches = [
        cls for cls, pattern in CLASS_KEYWORDS.items()
        if re.search(pattern, caption_text, flags=re.IGNORECASE)
    ]
    return matches if matches else ["other"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, default=Path("data_cache/annotations_processed_150.json"))
    parser.add_argument("--clip-ids", type=Path, default=Path("data_cache/small150_128shape.clip_ids.txt"))
    parser.add_argument("--output", type=Path, default=Path("data_cache/clip_semantic_classes.json"))
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text())
    clip_ids = [line.strip() for line in args.clip_ids.read_text().splitlines() if line.strip()]

    missing = [c for c in clip_ids if c not in annotations]
    if missing:
        raise RuntimeError(f"{len(missing)} clip_ids have no caption entry: {missing[:10]}")

    result = {}
    audit_rows = []
    class_counts = Counter()
    multi_match_count = 0
    for clip_id in clip_ids:
        captions = [a["text"] for a in annotations[clip_id]["annotations"]]
        joined = " ".join(captions)
        classes = classify_clip(joined)
        result[clip_id] = classes
        for c in classes:
            class_counts[c] += 1
        if len(classes) > 1:
            multi_match_count += 1
        audit_rows.append((clip_id, classes, captions))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))

    audit_path = args.output.with_suffix(".audit.txt")
    with open(audit_path, "w") as f:
        for clip_id, classes, captions in audit_rows:
            f.write(f"{clip_id}  ->  {classes}\n")
            for cap in captions:
                f.write(f"    - {cap}\n")
            f.write("\n")

    print(f"Classified {len(clip_ids)} clips -> {args.output}")
    print(f"Audit table (for human review) -> {audit_path}")
    print(f"\nMulti-class clips: {multi_match_count}/{len(clip_ids)} "
          f"({100 * multi_match_count / len(clip_ids):.1f}%)")
    print("\nPer-class counts (assign-to-all, so these sum to more than 150 if any clip multi-matches):")
    for cls in list(CLASS_KEYWORDS.keys()) + ["other"]:
        print(f"  {cls:10s}: {class_counts.get(cls, 0)}")


if __name__ == "__main__":
    main()
