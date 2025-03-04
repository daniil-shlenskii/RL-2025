from gymnasium.envs.registration import register

from .cardenv_1d import CardEnv
from .obstacles import Simple1DPathEnv
from .point import PointEnv
from .trex_env_simplified import TRexEnvSimplified

register(
    id="TRex-v0",
    entry_point="src.envs.custom_envs.trex_env:TRexEnv",
    max_episode_steps=1000,
)
