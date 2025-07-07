import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EnergyDataset(Dataset):
    def __init__(self, energy_path: str, price_path: str, forecast_size: int, customer: int, mode: str):
        csv_data = pd.read_csv(energy_path, header=0)
        csv_data['Date'] = pd.to_datetime(csv_data['Date'])
        csv_data['Price'] = pd.read_csv(price_path, header=0)['Price']
        csv_data.fillna(0, inplace=True)
        self._batteryCapacity = self._calcBatteryCapacity(csv_data.loc[0:17519].reset_index(drop=True), customer)
        if mode == 'train':
            csv_data = csv_data.loc[0:17519].reset_index(drop=True)
        elif mode == 'eval':
            csv_data = (csv_data.loc[17520:35039]).reset_index(drop=True)
        elif mode == 'test':
            csv_data = (csv_data.loc[35087:52559]).reset_index(drop=True)
        dates = csv_data['Date']
        fractional_hours = dates.dt.hour + dates.dt.minute / 60.0
        hour_sin = np.sin(2 * np.pi * fractional_hours / 24)
        hour_cos = np.cos(2 * np.pi * fractional_hours / 24)

        self._data = pd.DataFrame({'prosumption': csv_data[f'prosumption_{customer}'],
                                   'load': csv_data[f'load_{customer}'],
                                   'pv': csv_data[f'pv_{customer}'],
                                   'price': csv_data['Price'],
                                   'sin': hour_sin,
                                   'cos': hour_cos})
        self._forecast_size = forecast_size
     
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        idx = int(idx)
        prosumption = torch.tensor(self._data['prosumption'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
        price = torch.tensor(self._data['price'].loc[idx:idx+self._forecast_size].values, dtype=torch.float32)
        feature = np.array([self._data['sin'].loc[idx:idx+self._forecast_size].to_numpy(),self._data['cos'].loc[idx:idx+self._forecast_size].to_numpy()])
        feature = torch.tensor(feature, dtype=torch.float32).transpose(0,1)
        return prosumption, price, feature
    
    def _calcBatteryCapacity(self, csv_data_train, customer):
        prosumption = torch.tensor(csv_data_train[f'prosumption_{customer}'].values)
        daily_values = prosumption.view(365, 48)
        daily_negative_sums = daily_values.where(daily_values < 0, torch.zeros_like(daily_values)).sum(dim=1)
        return torch.ceil(torch.abs(daily_negative_sums.mean()))
        
    def getBatteryCapacity(self):
        return self._batteryCapacity.float()