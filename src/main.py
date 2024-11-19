import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
import gymnasium as gym
import envs.register_env as register_env
from utils.EnergyDataset import EnergyDataset


cs = ConfigStore.instance()
cs.store(name="hydra_config", node=HydraConfig)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    train_dataset = EnergyDataset("data/Final_Energy_dataset.csv",'train')
    env = gym.make('EnergyManagementEnv-v0', customer=1, dataset=train_dataset, cfg=cfg.env_params)

    state = env.reset()
    done = False

    while(not done):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        if done:
            state = env.reset()
        print("It works!")
    return

if __name__ == "__main__":
    main()