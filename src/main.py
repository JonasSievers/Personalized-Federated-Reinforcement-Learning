import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
import gymnasium as gym
import envs.register_env as register_env


cs = ConfigStore.instance()
cs.store(name="hydra_config", node=HydraConfig)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    env = gym.make('EnergyManagementEnv-v0', cfg=cfg.env_params)
    state = env.reset()
    
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        if done:
            print("done")
            state = env.reset()
    return

if __name__ == "__main__":
    main()