# Shape Embedding + Concat: Replacement for FiLM

Replaces the FiLM conditioning mechanism with a simpler, more stable design.
Motivated by the two failure modes documented in `README.film-fail.md`.

## Architecture

```
morphology_obs (11-dim: gender + 10 betas)
    → normalize (gender kept as-is, betas / 3.0)
    → Linear(→ 64) → SiLU
    → [optional: Linear(→ 64) → SiLU]        ← ablate: 1 vs 2 layers
    → shape_embed  (64-dim)
                        │
[max_coords_obs ]       │
[mimic_target_poses] ──cat──→  normalized flat obs  →  standard MLP trunk  →  output
[previous_actions ]
```

The last hidden layer's width implicitly defines the embedding dimension —
no separate config field needed.

## Why this is better than FiLM

| Property | FiLM | Shape Embed + Concat |
|---|---|---|
| Conditioner output size | 2 × layers × hidden (up to 12,288) | embed_dim (64) |
| Trunk coupling | Multiplicative (gamma × h + beta) | Additive (concat) |
| Gradient stability | Trunk grads scaled by gamma | Trunk grads unaffected |
| Conditioner trainability | Hard (massive fanout) | Easy (small projection) |

## Why this is better than raw concat

Raw concat feeds the 11-dim morphology vector directly alongside 400–600+ dim
main obs. The trunk must learn all shape reasoning implicitly from raw float values.

A small nonlinear projector lets the model learn a compact, nonlinear shape basis
(e.g., capturing body proportions, limb ratios) before the representation reaches
the trunk. This is analogous to a learned positional encoding: same information,
more usable form.

## Configuration knobs

- `cond_hidden_units`: width of each encoder layer, e.g. `[64]` (1 layer) or `[64, 64]` (2 layers)
- `cond_activation`: activation inside the encoder (default `silu`)
- `beta_norm_scale`: betas are divided by this before encoding (default `3.0`, matching FiLM)
- Everything else (trunk depth/width, normalization, output activation) is identical to `MLPWithConcat`

## Files

| File | Role |
|---|---|
| `protomotions/agents/common/shape_embed_mlp.py` | `ShapeEmbedMLPConfig` + `ShapeEmbedMLP` module |
| `examples/experiments/mimic/mlp_shape_embed.py` | Experiment config (drop-in replacement for `mlp_film.py`) |

## Ablation plan

1. `mlp.py` — baseline: raw concat of morphology_obs into flat obs
2. `mlp_shape_embed.py` with `cond_hidden_units=[64]` — 1-layer encoder
3. `mlp_shape_embed.py` with `cond_hidden_units=[64, 64]` — 2-layer encoder

Goal: determine whether the nonlinear projection adds value over raw concat,
and whether FiLM's poor performance was architectural rather than a capacity issue.
