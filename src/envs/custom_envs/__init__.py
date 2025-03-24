from gymnasium.envs.registration import register

register(
    id="TRex1dEnv-v0",
    entry_point="src.envs.custom_envs.trex_1d_env:TRex1dEnv",
)

register(
    id="TRexJumpEnv-v0",
    entry_point="src.envs.custom_envs.trex_jump_env:TRexJumpEnv",
)

register(
    id="TRexJumpSquatEnv-v0",
    entry_point="src.envs.custom_envs.trex_jumps_quat_env:TRexJumpSquatEnv",
)
