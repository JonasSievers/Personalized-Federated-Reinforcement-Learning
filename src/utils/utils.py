from torchrl.data import Bounded, Unbounded, Composite
from torchrl.envs.transforms import TransformedEnv, InitTracker
import torch
from envs.EnergyManagementEnv import EnergyManagementEnv
from utils.EnergyDataset import EnergyDataset
from utils.Networks import CustomActorNet, CustomCriticNet


def createEnvSpecs(cfg):
    observation_spec = Composite(observation=Unbounded(shape=(2 * cfg.env.forecast_horizon + 3,), dtype=torch.float32))
    action_spec = Bounded(low=-cfg.env.power_battery/2, high= cfg.env.power_battery/2, shape=(1,), dtype=torch.float32)
    return observation_spec, action_spec

def create_envs(customer, datasets, cfg, device):
    return [TransformedEnv(EnergyManagementEnv(customer=customer, dataset=dataset, cfg=cfg.env, device=device),InitTracker()) for dataset in datasets]

def createDatasets(cfg):
    return [EnergyDataset(data_path=cfg.dataset_path,mode=mode) for mode in ['test', 'eval', 'train']]

def createActorCritic(cfg):
    observation_spec, action_spec = createEnvSpecs(cfg)
    if cfg.networks.description=='nn':
        actor_net = CustomActorNet(observation_spec=observation_spec.clone(),
                                       action_spec=action_spec.clone(),
                                       fc_layers=cfg.networks.actor.fc_layers)
        critic_net = CustomCriticNet(observation_spec=observation_spec.clone(),
                                        action_spec=action_spec.clone(),
                                        obs_fc_layers=cfg.networks.critic.obs_fc_layers,
                                        joint_fc_layers=cfg.networks.critic.joint_fc_layers)
    else:
        actor_net = None
        critic_net = None
    return actor_net, critic_net

