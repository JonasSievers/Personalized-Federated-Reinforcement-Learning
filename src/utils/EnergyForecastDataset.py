import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

from utils.EnergyDataset import EnergyDataset
from utils.Forecaster import Forecaster

class EnergyForecastDataset(Dataset):
    def __init__(self, energy_path: str, price_path: str, forecast_size: int, customer: int, mode: str, time_feature: bool):
        self._forecast_size = forecast_size
        self._forecaster = Forecaster(cfg=None, customer=customer, mode='predict', calc_metric=False)
        self._ds = EnergyDataset(energy_path, price_path, forecast_size, customer, mode, time_feature)
        self._dl = iter(DataLoader(dataset=self._ds, batch_size=1, shuffle=False, num_workers=0))
        forecast = []
        for batch in self._dl:
            net_load, _, features = batch
            forecast.append(self._forecaster.predict(net_load[:,:48], features[:,:48,:]))
        self._net_load = torch.tensor(forecast)
    
    def __len__(self):
        return len(self._ds)-self._forecast_size
    
    def __getitem__(self, idx):
        net_load, price, _ = self._ds[idx+self._forecast_size]
        net_load_forecast = self._net_load[idx:idx+self._forecast_size]
        cat__net_load = torch.cat((net_load[:1], net_load_forecast), dim=0)
        return cat__net_load, price

