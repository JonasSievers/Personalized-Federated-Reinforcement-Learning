from torchrl.data import Bounded, Unbounded, Composite
from torchrl.envs.transforms import TransformedEnv, InitTracker
import torch
from envs.EnergyManagementEnv import EnergyManagementEnv
from utils.EnergyDataset import EnergyDataset
from utils.Networks import CustomActorNet, CustomCriticNet


def createEnvSpecs(battery_spec, cfg):
    observation_spec = Composite(observation=Unbounded(shape=(2 * cfg.env.forecast_horizon + 3,), dtype=torch.float32))
    action_spec = Bounded(low=-battery_spec/2, high= battery_spec/2, shape=(1,), dtype=torch.float32)
    return observation_spec, action_spec

def create_envs(customer, datasets, battery_spec, cfg, device):
    return [TransformedEnv(EnergyManagementEnv(customer=customer, dataset=dataset, battery_spec=battery_spec, cfg=cfg.env, device=device),InitTracker()) for dataset in datasets]

def createDatasets(cfg):
    return [EnergyDataset(energy_path=cfg.energy_dataset_path, price_path=cfg.price_dataset_path,mode=mode) for mode in ['train', 'eval', 'test']]

def calcBatteryCapacity(customer, dataset):
    output = {}
    for cust in customer:
        net_load = dataset.getTensor(customer=cust, idx_start=None, idx_end=None)
        daily_values = net_load.view(365, 48)
        daily_negative_sums = daily_values.where(daily_values < 0, torch.zeros_like(daily_values)).sum(dim=1)
        output[cust] = torch.ceil(torch.abs(daily_negative_sums.mean()))
    return output
