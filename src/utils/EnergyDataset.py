import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EnergyDataset(Dataset):
    def __init__(self, energy_path: str, price_path: str, forecast_size: int, customer: int, mode: str, time_feature: bool):
        csv_data = pd.read_csv(energy_path, header=0)
        csv_data['Date'] = pd.to_datetime(csv_data['Date'])
        csv_data['Price'] = pd.read_csv(price_path, header=0)['Price']
        csv_data.fillna(0, inplace=True)
        if mode == 'train':
            csv_data = csv_data.loc[0:17519].reset_index(drop=True)
        elif mode == 'eval':
            csv_data = (csv_data.loc[17520:35039]).reset_index(drop=True)
        elif mode == 'test':
            csv_data = (csv_data.loc[35040:52559]).reset_index(drop=True)
        self._data = pd.DataFrame({'prosumption': csv_data[f'prosumption_{customer}'],
                                   'load': csv_data[f'load_{customer}'],
                                   'pv': csv_data[f'pv_{customer}'],
                                   'price': csv_data['Price']})
        if time_feature:
            dates = csv_data['Date']
            fractional_hours = dates.dt.hour + dates.dt.minute / 60.0
            hour_sin = np.sin(2 * np.pi * fractional_hours / 24)
            hour_cos = np.cos(2 * np.pi * fractional_hours / 24)
            self._data['hour_sin'] = hour_sin
            self._data['hour_cos'] = hour_cos
        self._forecast_size = forecast_size
        self._time_feature = time_feature
    
    def __len__(self):
        return len(self._data)-self._forecast_size
    
    def __getitem__(self, idx):
        net_load = torch.tensor(self._data['prosumption'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
        price = torch.tensor(self._data['price'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
        if self._time_feature:
            hour_sin = torch.tensor(self._data['hour_sin'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
            hour_cos = torch.tensor(self._data['hour_cos'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
            features = torch.stack([hour_sin, hour_cos], dim=1)
            return net_load, price, features
        return net_load, price
    
    def getAllLoad(self):
        return torch.tensor(self._data['prosumption'].values, dtype=torch.float32)
    
    def getAllPrice(self):
        return torch.tensor(self._data['price'].values, dtype=torch.float32)

    # def getTensor(self, customer: int, idx_start: int = None, idx_end: int = None):
    #     if (idx_start is None) and (idx_end is None) :
    #         sample = self._data[f'net_load_{customer}']
    #         return torch.tensor(sample.values, dtype=torch.float32)
    #     elif idx_end is None:
    #         sample = self._data[['price', f'net_load_{customer}']].loc[idx_start]
    #         return torch.tensor(sample.values, dtype=torch.float32)
    #     else: 
    #         sample = self._data[['price', f'net_load_{customer}']].loc[idx_start:idx_end]
    #         return torch.tensor(sample.values, dtype=torch.float32).permute(1,0)