"""Task registration for the balance bot."""

from mjlab.tasks.registry import register_mjlab_task

from balance_bot.tasks.env_cfg import balance_bot_env_cfg, balance_bot_flat_env_cfg
from balance_bot.tasks.rl_cfg import balance_bot_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Drive-BalanceBot",
  env_cfg=balance_bot_env_cfg(),
  play_env_cfg=balance_bot_env_cfg(play=True),
  rl_cfg=balance_bot_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Drive-BalanceBot-Flat",
  env_cfg=balance_bot_flat_env_cfg(),
  play_env_cfg=balance_bot_flat_env_cfg(play=True),
  rl_cfg=balance_bot_ppo_runner_cfg(),
)
