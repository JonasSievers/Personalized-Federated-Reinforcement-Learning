import torch
from torch.utils.data import Dataset
import pandas as pd

class EnergyDataset(Dataset):
    def __init__(self, data_path: str, mode: str):
        csv_data = pd.read_csv(data_path, header=0)
        csv_data.drop(columns=['Date'], inplace=True)
        csv_data.set_index(pd.RangeIndex(0,52608),inplace=True)
        csv_data.fillna(0, inplace=True)
        if mode == 'train':
            csv_data = csv_data.loc[0:35087]
        elif mode == 'test':
            csv_data = csv_data.loc[35087:52608]
        else:
            csv_data = None
        self.data = csv_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return None

    def getTensor(self, customer: int, idx_start: int, idx_end: int = None):
        if idx_end is None:
            sample = self.data[['price', 'emissions', f'load_{customer}', f'pv_{customer}']].loc[idx_start]
            return torch.tensor(sample.values)
        else: 
            sample = self.data[['price', 'emissions', f'load_{customer}', f'pv_{customer}']].loc[idx_start:idx_end]
            return torch.tensor(sample.values).permute(1,0)