from torchrl.data import Bounded, Unbounded, Composite, Categorical
from torchrl.envs.transforms import TransformedEnv, InitTracker, ActionDiscretizer, Compose
import torch
from envs.EnergyManagementEnv import EnergyManagementEnv
from utils.EnergyForecastDataset import EnergyForecastDataset
from utils.EnergyDataset import EnergyDataset
from utils.CustomTransforms import CustomActionScaler,CustomObservationStandardization


def createEnvSpecs(cfg):
    observation_spec = Composite(observation=Unbounded(shape=(2 * (cfg.env.forecast_horizon+1) + 1,), dtype=torch.float32))
    action_spec = Bounded(low=-1, high=1, shape=(), dtype=torch.float32)
    return observation_spec, action_spec

def create_envs(customer, datasets, battery_cap, specs, cfg, device):
    # return [TransformedEnv(EnergyManagementEnv(customer=customer, dataset=dataset, battery_cap=battery_cap, specs=specs, cfg=cfg.env, device=device), InitTracker()) for dataset in datasets]
    # return [TransformedEnv(env=EnergyManagementEnv(customer=customer, 
    #                                                dataset=dataset, 
    #                                                battery_cap=battery_cap, 
    #                                                specs=specs, cfg=cfg.env, 
    #                                                device=device),
    #                         transform=Compose(InitTracker(),
    #                                           ActionDiscretizer(num_intervals=cfg.algorithm.num_intervals_discretized,
    #                                                             action_key = 'action', 
    #                                                             categorical=True, 
    #                                                             sampling=ActionDiscretizer.SamplingStrategy.MEDIAN))) for dataset in datasets]
    return [TransformedEnv(env=EnergyManagementEnv(customer=customer, 
                                                   dataset=dataset, 
                                                   battery_cap=battery_cap, 
                                                   specs=specs, cfg=cfg.env, 
                                                   device=device),
                            transform=Compose(InitTracker(),
                                              CustomActionScaler(action_spec=specs[1]))) for dataset in datasets]

def createDatasets(cfg, customer):
    ds_arr = [EnergyDataset(energy_path=cfg.data.energy_dataset_path, 
                          price_path=cfg.data.price_dataset_path, 
                          forecast_size=cfg.env.forecast_horizon,
                          customer=customer, 
                          mode=mode, 
                          time_feature=False) for mode in ['train', 'eval', 'test']]
    if cfg.use_forecast == True:
        forecast_ds = EnergyForecastDataset(energy_path=cfg.data.energy_dataset_path,
                                            price_path=cfg.data.price_dataset_path, 
                                            forecast_size=cfg.env.forecast_horizon,
                                            customer=customer, 
                                            mode='test', 
                                            time_feature=True)
        ds_arr.append(forecast_ds)
    return ds_arr

def calcBatteryCapacity(dataset):
    net_load = dataset.getAllLoad()
    daily_values = net_load.view(365, 48)
    daily_negative_sums = daily_values.where(daily_values < 0, torch.zeros_like(daily_values)).sum(dim=1)
    return torch.ceil(torch.abs(daily_negative_sums.mean()))
