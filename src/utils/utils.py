from torchrl.data import Bounded, Unbounded, Composite, Categorical
from torchrl.envs.transforms import TransformedEnv, InitTracker, ActionDiscretizer, Compose
import torch
from envs.EnergyManagementEnv import EnergyManagementEnv
from utils.EnergyDataset import EnergyDataset


def createEnvSpecs(battery_cap, cfg):
    battery_power = battery_cap/2
    observation_spec = Composite(observation=Unbounded(shape=(2 * cfg.env.forecast_horizon + 3,), dtype=torch.float32))
    action_spec = Bounded(low=-battery_power/2, high=battery_power/2, shape=(1,), dtype=torch.float32)
    return observation_spec, action_spec

def create_envs(customer, datasets, battery_cap, specs, cfg, device):
    return [TransformedEnv(EnergyManagementEnv(customer=customer, dataset=dataset, battery_cap=battery_cap, specs=specs, cfg=cfg.env, device=device), InitTracker()) for dataset in datasets]
    # return [TransformedEnv(EnergyManagementEnv(customer=customer, dataset=dataset, battery_cap=battery_cap, specs=specs, cfg=cfg.env, device=device),
    #                        Compose(InitTracker(),ActionDiscretizer(num_intervals=torch.tensor([50]), out_action_key="action_disc", categorical=True, sampling=ActionDiscretizer.SamplingStrategy.MEDIAN))) for dataset in datasets]

def createDatasets(cfg, customer):
    return [EnergyDataset(energy_path=cfg.data.energy_dataset_path, price_path=cfg.data.price_dataset_path, forecast_size=cfg.env.forecast_horizon, customer=customer, mode=mode) for mode in ['train', 'eval', 'test']]

def calcBatteryCapacity(dataset):
    net_load = dataset.getAllLoad()
    daily_values = net_load.view(365, 48)
    daily_negative_sums = daily_values.where(daily_values < 0, torch.zeros_like(daily_values)).sum(dim=1)
    return torch.ceil(torch.abs(daily_negative_sums.mean()))
