"""rsl_rl PPO runner config with a recurrent (LSTM) actor-critic.

rsl_rl's `RNNModel` runs an RNN directly on raw actor observations,
followed by an MLP (`hidden_dims`) that acts as the "decoder" mapping the
LSTM's hidden state down to actions (Variant A from the network-size
discussion -- see /memories/session/plan.md -- chosen over a custom
encoder-decoder-then-tiny-LSTM layout because it needs no custom rsl_rl
model class). Field names below follow the installed `mjlab.rl` /
`rsl_rl_lib` API (`RslRlOnPolicyRunnerCfg` / `RslRlModelCfg`).
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# Must match tools/quantize_export network spec (single LSTM layer, hidden
# size chosen small enough to run in real time on an ARMv5TEJ CPU w/o FPU).
LSTM_HIDDEN_SIZE = 96
# Decoder width after the LSTM (= tools.quantize_export.network_spec.DECODER_DIM).
DECODER_HIDDEN_SIZE = 96


def RoboCupPpoRunnerCfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for the RoboCup kick task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name="RNNModel",
            hidden_dims=(DECODER_HIDDEN_SIZE,),
            activation="tanh",  # Match embedded-nn's supported Tanh kernel.
            obs_normalization=True,
            rnn_type="lstm",
            rnn_hidden_dim=LSTM_HIDDEN_SIZE,
            rnn_num_layers=1,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name="RNNModel",
            hidden_dims=(64,),
            activation="tanh",
            obs_normalization=True,
            rnn_type="lstm",
            rnn_hidden_dim=LSTM_HIDDEN_SIZE,
            rnn_num_layers=1,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="ev3_robocup_kick",
        logger="tensorboard",
        num_steps_per_env=24,
        max_iterations=6000,
        save_interval=100,
    )