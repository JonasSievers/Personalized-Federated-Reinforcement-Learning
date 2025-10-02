import torch
from tensordict import TensorDict
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class OnlineDataset(Dataset):
    def __init__(self, data_path: str, sliding_window_size: int, sliding_window_offset: int, forecast_size: int, building_id: int, mode: str, device):

        data = torch.load(f'{data_path}/{building_id}.pt', weights_only=False)
        data['price'] = torch.load(f'{data_path}/price.pt', weights_only=False)

        self._forecast_size = forecast_size

        if sliding_window_size != sliding_window_offset:
            overlap_correction = torch.max(torch.tensor(1, dtype=torch.int32), (torch.tensor(sliding_window_size/sliding_window_offset)).int() - 1)
        else:
            overlap_correction = 0

        # 1038 days for ddpg training (1037 clean days)
        if mode == 'train':
            selected_data = data[0:49824]
            max_data = torch.tensor((49824-forecast_size)/sliding_window_offset).int() - overlap_correction
        # 29 days for validation (28 clean days)
        elif mode == 'val':
            selected_data = data[49824:51216]
            max_data = torch.tensor((1392-forecast_size)/sliding_window_offset).int() - overlap_correction
        # 29 days for testing (28 clean days)
        elif mode == 'test':
            selected_data = data[51216:52608]
            max_data = torch.tensor((1392-forecast_size)/sliding_window_offset).int() - overlap_correction       
            
        self._data = TensorDict({
                'load': selected_data['load'].unfold(dimension=0,size=sliding_window_size+forecast_size,step=sliding_window_offset)[0:max_data],
                'pv': selected_data['pv'].unfold(dimension=0,size=sliding_window_size+forecast_size,step=sliding_window_offset)[0:max_data],
                'prosumption': selected_data['prosumption'].unfold(dimension=0,size=sliding_window_size+forecast_size,step=sliding_window_offset)[0:max_data],
                'price': selected_data['price'].unfold(dimension=0,size=sliding_window_size+forecast_size,step=sliding_window_offset)[0:max_data]
            }, 
            batch_size=torch.Size([max_data, sliding_window_size+forecast_size]),
            device=device)

        self._batteryCapacity = self._calcBatteryCapacity(data[0:49824]['prosumption'])
     
    def __len__(self):
        return self._data.batch_size[0]
    
    def __getitem__(self, idx):
        return self._data[idx]
    
    def _calcBatteryCapacity(self, tensor):
        daily_values = tensor.view(-1, 48)
        daily_negative_sums = daily_values.where(daily_values < 0, torch.zeros_like(daily_values)).sum(dim=1)
        return torch.ceil(torch.abs(daily_negative_sums.mean()))
        
    def getBatteryCapacity(self):
        return self._batteryCapacity.float()