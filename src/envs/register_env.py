from gymnasium.envs.registration import register

register(
    id="EnergyManagementEnv-v0",
    entry_point="envs.EnergyManagementEnv:EnergyManagementEnv",
    kwargs= {"cfg": None}
)