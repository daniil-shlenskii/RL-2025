from gymnasium.envs.registration import register

register(
    id="TRexJump-v0",
    entry_point="src.envs.custom_envs.trex_jump_env:TRexJump",
)

register(
    id="TRex-v0",
    entry_point="src.envs.custom_envs.trex_jumpsquat_env:TRexEnv",
    max_episode_steps=1000,
)
