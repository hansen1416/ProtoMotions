# Capability-table source audit

Checked 10 October 2026. The table is published capability/evidence, not a performance ranking.

| System | Primary source and location | Supported entry / boundary |
|---|---|---|
| PHC (ICCV 2023, arXiv v3) | https://arxiv.org/html/2305.06456v3 — §4 implementation/data; Appendix B.1 humanoid construction and training | 11,313 training / 140 test AMASS sequences; shape-varying bodies and sampled-shape training described; quantitative evaluations use mean neutral SMPL body, with qualitative variation. Do not label PHC fixed-body-only. |
| ASE (TOG 2022, arXiv v2) | https://arxiv.org/html/2205.01906v2 — §4, §8, §10.1 | Latent reusable skills, rather than exact clip tracking; 187 clips, sword/shield humanoid; no systematic unseen-body clip-tracking test reported. |
| ProtoMotions 3 | https://github.com/NVlabs/ProtoMotions — official README, accessed 10 October 2026 | Framework supports tracking, distributed training, robot/character configurations, retargeting and generative policies. Different configured robots do not establish a shared multi-shape policy. README claims are software documentation, not a matched paper benchmark; current v3 documentation is not the frozen experiment revision. |
| Won & Lee (TOG 2019) | https://mrl.snu.ac.kr/publications/ProjectMorphCon/MorphCon.pdf — abstract / introduction | Parametric body control, varying height/weight/proportions, novel and changing shapes. Direct prior art for body-conditioned physical control; distinct experimental task. |
| HUMOS (ECCV 2024) | https://humos.is.tue.mpg.de/ and https://arxiv.org/abs/2409.03944 | Shape-conditioned motion generation with physical consistency objectives; inherited pretrained reference generator, not our online tracking controller. |
| This study | evidence/unseen_shapes_v2/REPORT.md and manuscript test/reference/architecture tables | Paired corpus, identity-preserving workflow, evaluated temporal design and systematic new replay. No external matched performance run, causal conditioning test or all-body held-out-motion test. |

Not reported means absent from the cited evaluation, not unsupported in principle.
All body/clip/horizon/reference/success conventions must match before quantitative
performance scores can be presented as directly comparable.

The new unseen-body report calls zero-termination errors silent drift and says
none are falls. The manuscript deliberately uses the narrower observable
statement: error-threshold failures without recorded termination. A termination
flag depends on configuration; an independent fall/rollout audit is missing.
