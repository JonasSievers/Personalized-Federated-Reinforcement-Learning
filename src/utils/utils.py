from torchrl.data import Bounded, Unbounded, Composite, Categorical
from torchrl.envs import (
    CatTensors,
    TransformedEnv,
    UnsqueezeTransform,
    Compose,
    InitTracker,
)
import torch
from envs.Environment import BatteryScheduling
from utils.EnergyForecastDataset import EnergyForecastDataset
from utils.EnergyDataset import EnergyDataset
from utils.CustomTransforms import CustomActionScaler,CustomObservationStandardization


def make_specs(cfg):
    observation_spec = Composite(observation=Unbounded(shape=(2 * (cfg.env.forecast_horizon+1) + 1,), dtype=torch.float32))
    action_spec = Bounded(low=-1, high=1, shape=(), dtype=torch.float32)
    return observation_spec, action_spec

def make_env(cfg, datasets, device):
    return [TransformedEnv(env=BatteryScheduling(cfg=cfg,
                                                 dataset=dataset,
                                                 eval = eval,
                                                 device=device),
                            transform=Compose(InitTracker(),
                                              CustomActionScaler(max_action=dataset.getBatteryCapacity()/4),
                                             
                                            # CustomObservationStandardization(battery_cap=dataset.getBatteryCapacity()),
                                             UnsqueezeTransform(dim=-1,
                                                                 in_keys=['soe', 'prosumption', 'price', 'cost'],
                                                                 in_keys_inv=['soe', 'prosumption', 'price', 'cost']),
                                            CatTensors(dim=-1,
                                                         in_keys=['time_feature', 'soe', 'prosumption','prosumption_forecast','price','price_forecast'],
                                                         out_key='observation',
                                                         del_keys=False)
                                            )) for dataset, eval in zip(datasets,[False,True,True])]

def make_datasets(cfg, customer):
    ds_arr = [EnergyDataset(energy_path=cfg.data.energy_dataset_path, 
                          price_path=cfg.data.price_dataset_path, 
                          forecast_size=cfg.env.forecast_horizon,
                          customer=customer, 
                          mode=mode) for mode in ['train', 'eval', 'test']]
    # TODO Currently no forecast option avaible
    # if cfg.use_forecast == True:
    #     forecast_ds = EnergyForecastDataset(energy_path=cfg.data.energy_dataset_path,
    #                                         price_path=cfg.data.price_dataset_path, 
    #                                         forecast_size=cfg.env.forecast_horizon,
    #                                         customer=customer, 
    #                                         mode='test', 
    #                                         time_feature=True)
    #     ds_arr.append(forecast_ds)
    return ds_arr
