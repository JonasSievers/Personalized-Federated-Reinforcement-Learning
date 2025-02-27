import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EnergyDataset(Dataset):
    def __init__(self, energy_path: str, price_path: str, forecast_size: int, customer: int, mode: str):
        csv_data = pd.read_csv(energy_path, header=0)
        csv_data['Date'] = pd.to_datetime(csv_data['Date'])
        csv_data = csv_data[csv_data['Date'].dt.date !=pd.to_datetime('2012-02-29').date()].reset_index(drop=True)
        csv_data['price'] = pd.read_csv(price_path, header=0) /100
        csv_data.fillna(0, inplace=True)
        if mode == 'train':
            csv_data = csv_data.loc[0:17519].reset_index(drop=True)
        elif mode == 'eval':
            csv_data = (csv_data.loc[17520:35039]).reset_index(drop=True)
        elif mode == 'test':
            csv_data = (csv_data.loc[35040:52559]).reset_index(drop=True)
        self._data = pd.DataFrame({'net_load': csv_data[f'load_{customer}'] - csv_data[f'pv_{customer}']})
        self._data['price'] = csv_data['price']
        self._forecast_size = forecast_size
    
    def __len__(self):
        return len(self._data)-self._forecast_size
    
    def __getitem__(self, idx):
        net_load = torch.tensor(self._data['net_load'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
        price = torch.tensor(self._data['price'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
        return net_load, price
    
    def getAllLoad(self):
        return torch.tensor(self._data['net_load'].values, dtype=torch.float32)

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