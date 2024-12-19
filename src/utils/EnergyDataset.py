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
            csv_data = csv_data.loc[0:17519]
        elif mode == 'eval':
            csv_data = (csv_data.loc[17520:35087]).set_index(pd.RangeIndex(0,17568))
        elif mode == 'test':
            csv_data = (csv_data.loc[35088:52607]).set_index(pd.RangeIndex(0,17520))
        self._data = pd.DataFrame({
            f'net_load_{i+1}': csv_data[f'load_{i+1}'] - csv_data[f'pv_{i+1}'] 
            for i in range(300)})
        self._data['price'] = csv_data['price']
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return None
    
    def getAll(self):
        return self._data

    def getTensor(self, customer: int, idx_start: int, idx_end: int = None):
        if idx_end is None:
            sample = self._data[['price', f'net_load_{customer}']].loc[idx_start]
            return torch.tensor(sample.values, dtype=torch.float32)
        else: 
            sample = self._data[['price', f'net_load_{customer}']].loc[idx_start:idx_end]
            return torch.tensor(sample.values, dtype=torch.float32).permute(1,0)