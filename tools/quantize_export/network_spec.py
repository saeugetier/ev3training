"""Single source of truth for the deployed policy network topology.

Both the Python export pipeline and the Rust firmware must agree on these
dimensions. If you change the mjlab agent cfg's `actor_hidden_dims` /
`rnn_hidden_size` (sim/mjlab_robocup/agents/rsl_rl_ppo_cfg.py) or the
observation terms (sim/mjlab_robocup/robocup_env_cfg.py), update this file
and firmware/src/policy.rs together.

Topology (Variant A: memory operates directly on raw observations, matching
rsl_rl's stock `ActorCriticRecurrent` -- RNN(obs) -> MLP decoder -> action;
no separate pre-LSTM encoder, since that would need a custom rsl_rl actor
class. See /memories/session/plan.md for the size trade-off discussion vs.
the alternative "encoder -> decoder -> tiny LSTM" layout):

    obs -> LSTM(INPUT_DIM -> LSTM_HIDDEN_DIM)   [gates use split Wx/Wh,
                                                  see reference_model.py]
        -> Dense(LSTM_HIDDEN_DIM -> DECODER_DIM) -> Tanh   [decoder]
        -> Dense(DECODER_DIM -> OUTPUT_DIM) -> Tanh        [action head]
"""

from __future__ import annotations

DEPTH_DIM = 64  # VL53L8CX 8x8 zones.
YAW_SINCOS_DIM = 2
YAW_RATE_DIM = 1
WHEEL_DELTAS_DIM = 2
LAST_ACTION_DIM = 3

INPUT_DIM = DEPTH_DIM + YAW_SINCOS_DIM + YAW_RATE_DIM + WHEEL_DELTAS_DIM + LAST_ACTION_DIM  # 72
LSTM_HIDDEN_DIM = 32
DECODER_DIM = 32
OUTPUT_DIM = 3  # wheel_left, wheel_right, kick_trigger

assert INPUT_DIM == 72
