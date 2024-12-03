from torchrl.data import Bounded, Unbounded, Composite
from torchrl.envs.transforms import TransformedEnv, InitTracker
import torch
from envs.EnergyManagementEnv import EnergyManagementEnv
from utils.EnergyDataset import EnergyDataset


def createEnvSpecs(env_params):
    observation_spec = Composite(observation=Unbounded(shape=(2 * env_params.forecast_horizon + 3,), dtype=torch.float32))
    action_spec = Bounded(low=-env_params.power_battery/2, high=env_params.power_battery/2, shape=(1,), dtype=torch.float32)
    return observation_spec, action_spec

def create_envs(customer, datasets, cfg):
    return [TransformedEnv(EnergyManagementEnv(customer=customer, dataset=dataset, cfg=cfg.experiment.env_params), InitTracker()) for dataset in datasets]

def createDatasets(cfg):
    return [EnergyDataset(data_path=cfg.experiment.dataset_path,mode=mode) for mode in ['train', 'eval', 'test']]

