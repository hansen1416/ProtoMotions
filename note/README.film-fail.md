# Why FiLM Failed at Scale

FiLM (Feature-wise Linear Modulation) was tried as a morphology-conditioning mechanism
for the HHI multi-shape policy. It did not learn well. Two compounding issues explain why.

## Issue 1: Fanout bottleneck

The conditioner network is small (64→64 hidden units), but it must produce
`2 × num_layers × hidden_dim` output values — one gamma and one beta per hidden unit
per trunk layer.

For the actor (6 layers × 1024 units):

```
conditioner output size = 2 × 6 × 1024 = 12,288
```

A 64-unit network producing 12,288 values is a severe compression/expansion mismatch.
Gradients flowing back through `cond_linear` are diluted across all those outputs,
making the conditioner hard to train. Going from 1 motion to thousands of motions
worsens this because the conditioner must now cover a much wider distribution of body
shapes with the same tiny capacity.

## Issue 2: Multiplicative instability

FiLM modulates each trunk layer as:

```
h_l = h_l * gamma_l + beta_l
```

This means trunk gradients at layer `l` are scaled by `gamma_l`. If `gamma_l` drifts
far from 1.0 early in training — which is likely when the conditioner is poorly
initialized or under-trained — the effective learning rate for the trunk becomes
shape-dependent and unstable.

With a diverse motion dataset, the conditioner sees widely varying shapes per minibatch.
Noisy gamma estimates amplify this instability and prevent the trunk from converging
to a stable feature representation.

## Summary

| Problem | Root cause | Worsens with more motions? |
|---|---|---|
| Fanout bottleneck | 64-unit conditioner → 12,288 outputs | Yes — wider shape distribution |
| Multiplicative instability | Trunk gradients scaled by gamma | Yes — noisier gamma per minibatch |

The combination means FiLM is poorly suited as a scaling strategy when moving from
a single-motion to a thousands-of-motions training regime.

See `README.shape-embedding.md` for the replacement design.
