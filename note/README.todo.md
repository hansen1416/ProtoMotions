1. double check scripts/export_humos_to_amass_npz.py, I remember HUMOS performed better on smplx not smpl.
2. do visualize on multiple humanoid/moition in same window
    1. need to let protomotions load multiple smpl robot
    2. display multiple motion in same window


- multiple humanoid / robot loading
- multiple motions, one motion per robot / env
- env_id → gender/beta_key
    motion_id → gender/beta_key
    sample real AMP demo from the same gender/beta_key as the simulated humanoid
    ensure: fake motion from body shape m
            vs.
            real motion from the same body shape m

    MotionLib metadata:
        motion_id -> gender, beta_key, beta vector

    Env metadata:
        env_id -> humanoid asset, gender, beta_key, beta vector

    Sampler:
        sample_motion(env_id) only from matching shape bucket

    AMP demo sampler:
        sample_real_amp(env_id) only from matching shape bucket

- actor / critic observation expansion with gender + betas
- discriminator observation expansion or conditioning
- FiLM-conditioned MLP, where FiLM takes shape input
- shape-conditioned actor / critic / discriminator path

| Old `hhi` logic               | ProtoMotions3 target                              |
| ----------------------------- | ------------------------------------------------- |
| multiple humanoid XML loading | robot config + simulator asset creation           |
| `env_id_beta_keys_map`        | environment context / control component state     |
| HUMOS MotionLib metadata      | extend `protomotions/components/motion_lib.py`    |
| actor obs + betas             | observation component                             |
| critic FiLM                   | PPO / AMP model config + custom module            |
| discriminator FiLM            | AMP discriminator custom module                   |
| AMP demo shape matching       | AMP agent / replay / demo batch assembly          |
| reset to target motion frame  | mimic control component                           |
| PD scaling / residual PD      | robot config control + simulator action mapping   |
| jitter penalties              | reward component, e.g. smoothness / power penalty |

1. Data conversion
   HUMOS .pt / npz -> ProtoMotions .motion -> packaged MotionLib .pt

2. MotionLib extension
   store:
       motion_shape: [num_motions, 11]
       gender
       beta_key
       beta_key_motion_id_mapping

3. Multi-shape robot assets
   load correct SMPL/SMPL-X/MJCF asset per env
   verify joint limits, mass, inertia, contact geometry, height offset

4. Env context
   expose:
       env_shape
       env_gender
       env_beta_key
       env_motion_id

5. Observation component
   append or separately route morphology:
       obs = [base_obs, task_obs, gender, betas]

6. Model
   actor: FiLM or concat
   critic: FiLM or concat
   discriminator: FiLM strongly recommended
   disc critic: probably also shape-conditioned

7. AMP/demo sampling
   fake AMP obs uses env shape
   real AMP obs must be sampled from same shape bucket

8. Reset logic
   reset humanoid to reference motion frame from matching shape

9. Reward/termination
   tracking reward uses shape-matched reference
   contact reward uses correct body/contact indices
   power/smoothness penalties retained

10. Checkpoint/normalization
   old running mean/std cannot be blindly reused after obs shape changes
   load compatible layers only
   reinitialize new FiLM/shape branches

11. Validation
   single shape + single motion
   multi shape + single motion
   single shape + many motions
   multi shape + many motions
   AMP real/fake shape-match unit test