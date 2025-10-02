from torchrl.envs import (
    CatTensors,
    TransformedEnv,
    UnsqueezeTransform,
    Compose,
    InitTracker,
)
from environment.BatteryScheduling import BatteryScheduling
from dataset.OnlineDataset import OnlineDataset
import os


def make_dataset(cfg, mode, device):
    match mode:
        case 'train':
            ds = OnlineDataset(data_path=cfg.data_path,
                               sliding_window_size=cfg.dataset.sliding_window_size,
                               sliding_window_offset=cfg.dataset.sliding_window_offset,
                               forecast_size=cfg.dataset.forecast_horizon,
                               building_id=cfg.building_id,
                               mode=mode,
                               device=device)
        case 'val' | 'test':
            ds = OnlineDataset(data_path=cfg.data_path,
                                 sliding_window_size=1344,
                                 sliding_window_offset=1344,
                                 forecast_size=cfg.dataset.forecast_horizon,
                                 building_id=cfg.building_id,
                                 mode=mode,
                                 device=device)
        case _:
            ds = None
    return ds     

def make_env(cfg, dataset, device):
    return TransformedEnv(base_env=BatteryScheduling(cfg=cfg,
                                                datasets=dataset,
                                                device=device),
                            transform=Compose(InitTracker(),
                                                UnsqueezeTransform(dim=-1,
                                                                in_keys=['soe', 'prosumption', 'price', 'cost', 'step'],
                                                                in_keys_inv=['soe', 'prosumption', 'price', 'cost', 'step']),
                                                CatTensors(dim=-1,
                                                        in_keys=['soe', 'prosumption','prosumption_forecast','price','price_forecast'],
                                                        out_key='observation',
                                                        del_keys=False)).to(device=device)
                            ).to(device=device)
