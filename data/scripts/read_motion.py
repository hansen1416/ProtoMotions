"""
python data/scripts/read_motion.py /home/hlz/datasets/humos_proto/humos_8_offset.pt
Pretty-print the structure of a ProtoMotions .pt motion file.

Usage:
    python inspect_motion_pt.py <path_to_motion_file.pt> [--depth N] [--sample]

Options:
    --depth N     How many levels deep to recurse into nested dicts/lists (default: 4)
    --sample      Print a small sample of tensor values (first few elements)
"""

import sys
import argparse
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def tensor_summary(t, sample=False):
    """One-line description of a tensor."""
    info = f"Tensor {tuple(t.shape)}  dtype={t.dtype}  device={t.device}"
    if t.numel() > 0:
        try:
            info += f"  min={t.float().min().item():.4g}  max={t.float().max().item():.4g}"
        except Exception:
            pass
    if sample and t.numel() > 0:
        flat = t.reshape(-1)
        n = min(6, flat.numel())
        vals = [f"{flat[i].item():.4g}" for i in range(n)]
        ellipsis = "…" if flat.numel() > n else ""
        info += f"  sample=[{', '.join(vals)}{ellipsis}]"
    return info


def describe(obj, indent=0, depth=4, sample=False, max_list_items=6):
    prefix = "  " * indent
    if depth < 0:
        print(f"{prefix}... (max depth reached)")
        return

    if isinstance(obj, dict):
        print(f"{prefix}dict  ({len(obj)} keys)")
        for k, v in obj.items():
            print(f"{prefix}  [{repr(k)}]")
            describe(v, indent + 2, depth - 1, sample, max_list_items)

    elif isinstance(obj, (list, tuple)):
        kind = "list" if isinstance(obj, list) else "tuple"
        print(f"{prefix}{kind}  (len={len(obj)})")
        shown = min(len(obj), max_list_items)
        for i in range(shown):
            print(f"{prefix}  [{i}]")
            describe(obj[i], indent + 2, depth - 1, sample, max_list_items)
        if len(obj) > shown:
            print(f"{prefix}  ... ({len(obj) - shown} more items)")

    elif isinstance(obj, torch.Tensor):
        print(f"{prefix}{tensor_summary(obj, sample)}")

    elif hasattr(obj, "__dict__"):
        # dataclass / object
        cls = type(obj).__name__
        fields = vars(obj)
        print(f"{prefix}<{cls}>  ({len(fields)} attrs)")
        for k, v in fields.items():
            print(f"{prefix}  .{k}")
            describe(v, indent + 2, depth - 1, sample, max_list_items)

    else:
        print(f"{prefix}{type(obj).__name__}  =  {repr(obj)[:120]}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect a ProtoMotions .pt motion file.")
    parser.add_argument("path", help="Path to the .pt motion file")
    parser.add_argument("--depth", type=int, default=4, help="Recursion depth (default: 4)")
    parser.add_argument("--sample", action="store_true", help="Show tensor value samples")
    parser.add_argument("--max-list", type=int, default=6,
                        help="Max list/tuple items to show at each level (default: 6)")
    args = parser.parse_args()

    print(f"\nLoading: {args.path}\n{'─' * 60}")
    data = torch.load(args.path, map_location="cpu", weights_only=False)
    print(f"Top-level type: {type(data).__name__}\n{'─' * 60}\n")

    describe(data, indent=0, depth=args.depth, sample=args.sample, max_list_items=args.max_list)

    # ── bonus: if it looks like a MotionLib dict, print a tidy summary ──────
    if isinstance(data, dict):
        print(f"\n{'─' * 60}")
        print("Keys at top level:")
        for k in data.keys():
            print(f"  {repr(k)}")

        # Common ProtoMotions motion file patterns
        for count_key in ("n_motions", "num_motions", "motions"):
            if count_key in data:
                val = data[count_key]
                if isinstance(val, (int, float)):
                    print(f"\nNumber of motions ({count_key}): {val}")
                elif isinstance(val, (list, tuple)):
                    print(f"\nNumber of motions ({count_key}): {len(val)}")
                elif isinstance(val, torch.Tensor):
                    print(f"\n'{count_key}' tensor shape: {tuple(val.shape)}")
                break


if __name__ == "__main__":
    main()