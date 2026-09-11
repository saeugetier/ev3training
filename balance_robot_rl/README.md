# balance_bot_rl

Reinforcement-learning project for a **two-wheel self-balancing robot (segway)**,
built on [mjlab](https://github.com/mujocolab/mjlab) (Isaac-Lab-style manager API
on top of MuJoCo Warp).

The policy has to keep the inverted-pendulum chassis upright while following a
**two-button remote control** — one button per wheel: press both to drive
forward, press one to turn. Training happens on slightly uneven ground with
randomized friction, while light boxes are thrown at the robot.

Task IDs:

| Task | Ground | Disturbances |
| --- | --- | --- |
| `Mjlab-Drive-BalanceBot` | uneven, randomized friction | pushes + thrown boxes |
| `Mjlab-Drive-BalanceBot-Flat` | flat plane, fixed friction | pushes only |

## Quick start

```bash
uv sync                                   # install mjlab + this package
uv run list-envs                          # the balance bot tasks show up
uv run play Mjlab-Drive-BalanceBot --agent zero      # look at the robot
uv run python scripts/pd_baseline.py      # hand-tuned PD baseline (sanity check)
uv run train Mjlab-Drive-BalanceBot --env.scene.num-envs 4096
uv run play Mjlab-Drive-BalanceBot        # replay the latest checkpoint
```

Add `--agent.logger tensorboard` to `train` if you do not use Weights & Biases.

## Robot

`src/balance_bot/assets/balance_bot.xml` — a free-floating chassis with the
battery/electronics mass ~9 cm above the wheel axle, plus two hinge wheels
(radius 5 cm, 15 cm apart) driven by torque motors (peak 0.5 Nm each). The wheel
joints carry a damping term that models the DC motor back-EMF (stall torque over
a ~60 rad/s no-load speed), which also keeps the wheels from spinning up without
bound while slipping on low-friction ground.

## Sensors and actuators

The policy only consumes quantities that the real robot can measure:

| Signal | Source on the real robot | Observation term |
| --- | --- | --- |
| Tilt angle about the wheel axis | IMU/gyro, **zeroed at power-up** | `gyro_angle` |
| Angular rate (3 axes) | Gyro | `gyro_rate` |
| Wheel speed | Encoder differences | `wheel_speed` |
| Rotation-distance error per wheel | Encoder odometry vs. commanded rotation | `wheel_distance_error` |
| Remote control (left, right) | RC receiver | `remote_control` |
| Previous motor command | Controller state | `last_action` |

The actor group stacks the last 3 samples of each term (36 values) and is
corrupted with sensor noise during training. Because the gyro angle is zeroed
once at start-up, it carries a **constant per-episode offset** (±0.05 rad,
`NoiseModelWithAdditiveBiasCfg`) on top of per-sample noise, so the policy has to
be robust to a mis-zeroed IMU. The critic additionally sees privileged base
linear velocity and projected gravity.

Actions are normalized motor effort for the two wheels (`JointEffortActionCfg`,
one per wheel), applied at 100 Hz (2.5 ms physics timestep, `decimation=4`).
This mirrors the EV3, which has no on-device torque or speed servo — only
open-loop duty-cycle motor control (`set_duty_cycle_sp`) — so a policy output
of `-1..1` maps directly to a `-100..100` duty cycle on the real robot.

## Remote control command

`RemoteControlCommand` (`src/balance_bot/tasks/mdp.py`) resamples an independent
continuous control value in `[-1, 1]` for each wheel every 2–5 s. The values are
exposed to the policy as `remote_control` and scaled to wheel speed targets.

The term also integrates the commanded wheel rotation and compares it with the
encoder reading, giving a saturated odometry error that is both observed and
rewarded. Values near zero produce the station-keeping behavior.

## Rewards

| Term | Weight | Purpose |
| --- | --- | --- |
| `upright` | +2.0 | exponential kernel on tilt from vertical |
| `track_wheel_speed` | +1.5 | match the commanded wheel speeds |
| `track_wheel_distance` | +1.0 | odometry / station keeping |
| `alive` | +0.5 | survival bonus |
| `terminated` | −10.0 | penalty for falling |
| `action_rate` | −0.01 | smooth motor commands |
| `motor_torque` | −0.005 | energy |

Episodes end on a time-out (20 s), when the chassis tilts more than 0.6 rad, or
when `nan_detection` spots a diverged physics state. Resets randomize the start
pitch (±0.1 rad), yaw and wheel speeds. The observation groups use
`nan_policy="sanitize"` so a single bad physics frame resets the affected
environment instead of aborting the training run.

## Terrain, friction and disturbances

* **Uneven ground** — a procedural terrain grid (`uneven_terrain_cfg`) of 5×5
  patches: 25 % flat, 45 % random bumps of 0.5–2 cm at 5 cm resolution, 30 %
  long waves of 1–3.5 cm amplitude. Nothing dramatic, but enough that the wheels
  never roll on a perfect plane.
* **Varying friction** — the tangential friction of the ground geoms is resampled
  every episode from 0.25 to 1.2 (`dr.geom_friction`, `mode="reset"`, shared
  across all terrain geoms of an environment). The tyre friction in the MJCF is
  set to 0.2, below the lowest ground value, so MuJoCo's max() contact rule makes
  the randomized ground value the effective one.
* **Thrown boxes** — three 150 g, 8 cm cubes park around the robot. Every
  1.5–4 s `mdp.throw_box` teleports one of them to a random point 0.6–1.2 m away
  and 0.1–0.4 m above the chassis and gives it a ballistic velocity
  (2–5 m/s) aimed at the upper body. Measured effect on the PD baseline: peak
  tilt rises from 0.13 rad to 0.34 rad.
* **Pushes** — an additional velocity kick every 4–8 s.

Play mode keeps the terrain, friction randomization and boxes but drops the
velocity pushes.

## Layout

```
src/balance_bot/
  assets/balance_bot.xml   MJCF model (chassis, wheels, IMU + encoder sensors)
  robot.py                 EntityCfg for the robot
  boxes.py                 EntityCfgs for the light boxes
  tasks/mdp.py             remote-control command, observations, rewards, box throwing
  tasks/env_cfg.py         ManagerBasedRlEnvCfg, terrain and randomization
  tasks/rl_cfg.py          PPO hyperparameters
  tasks/__init__.py        task registration
scripts/pd_baseline.py     hand-tuned PD controller baseline
```

The package registers itself with mjlab through the `mjlab.tasks` entry point in
`pyproject.toml`, so `train`, `play` and `list-envs` pick the task up
automatically.
