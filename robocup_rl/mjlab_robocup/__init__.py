"""RoboCup dribble-and-kick task for mjlab, targeting EV3 deployment."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_robocup.agents.rsl_rl_ppo_cfg import RoboCupPpoRunnerCfg
from mjlab_robocup.robocup_env_cfg import make_robocup_env_cfg

TASK_ID = "Mjlab-RoboCup-Kick-v0"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_robocup_env_cfg(),
    play_env_cfg=make_robocup_env_cfg(play=True),
    rl_cfg=RoboCupPpoRunnerCfg(),
)
