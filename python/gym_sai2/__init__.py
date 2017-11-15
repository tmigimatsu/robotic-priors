from gym.envs.registration import register

register(
    id="sai2-v0",
    entry_point="gym_sai2.envs:SaiEnv"
)
