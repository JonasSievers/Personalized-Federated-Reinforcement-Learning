import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from utils.EnergyDataset import EnergyDataset

class Model(nn.Module):
    def __init__(self, input_shape):
        super(Model, self).__init__()

        input_size  = 48*3

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.01)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class Forecaster:
    def __init__(self, cfg, customer, mode, calc_metric=True):
        self._cfg = cfg
        self._customer = customer
        self._calc_metric = calc_metric
        self.results = {}
        self.metrics = {}
        self._mode = mode
        self._model = Model(input_shape=(16,48,3))
        if self._mode == 'train':
            self._dl_train = iter(DataLoader(dataset=EnergyDataset('data/Final_Energy_dataset.csv',
                                                                   'data/price.csv',
                                                                   48,
                                                                   customer,
                                                                   'train',
                                                                   True), 
                                            batch_size=16, 
                                            shuffle=True, 
                                            num_workers=0))
            self._dl_test = iter(DataLoader(dataset=EnergyDataset('data/Final_Energy_dataset.csv',
                                                                  'data/price.csv',
                                                                  48,
                                                                  customer,
                                                                  'test',
                                                                  True),
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0))
        elif self._mode == 'predict':
            self._model.load_state_dict(torch.load(f'models/forecaster/{customer}.pt'))
        

    def train(self, epochs=100):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        for epoch in range(epochs):
            self._model.train()
            for load, _, feature in self._dl_train:
                load_input = load[:,:48].unsqueeze(-1)
                load_target = load[:,48:]
                feature_input = feature[:,:48,:]
                input = torch.cat((load_input, feature_input), dim=2)
                target = load_target
                optimizer.zero_grad()
                output = self._model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        torch.save(self._model.state_dict(), f'models/forecaster/{self._customer}.pt')
        
    def evaluate(self):
        output = []
        target = []
        self._model.eval()
        for load, _, feature in self._dl_test:
            load_input = load[:,:48].unsqueeze(-1)
            load_target = load[:,48:]
            feature_input = feature[:,:48,:]
            input = torch.cat((load_input, feature_input), dim=2)
            target.append(load_target.item())
            output.append(self._model(input).detach().item())
        df = pd.DataFrame({'output': output, 'target': target})
        df.to_csv(f'{self._cfg.output_path}/forecaster/{self._customer}.csv', index=False)
        return target, output

    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return {"mse": mse, "rmse": rmse, "mae": mae}

    def predict(self, net_load, feature):
        self._model.eval()
        input = torch.cat((net_load.unsqueeze(-1), feature), dim=2)
        output = self._model(input)
        return output.item()