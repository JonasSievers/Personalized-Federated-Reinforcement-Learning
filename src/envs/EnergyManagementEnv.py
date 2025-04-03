import torch
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite
from tensordict import TensorDict, TensorDictBase

from utils.EnergyDataset import EnergyDataset

class EnergyManagementEnv(EnvBase):
    def __init__(self, customer, dataset, battery_cap, specs, cfg, device):
        super().__init__(device=device, batch_size=torch.Size([]))
        self._dtype = torch.float32
        self._customer = customer
        self._current_timestep = 0
        self._episode_ended = False
        self._timeslots_per_day = cfg.timeslots_per_day
        self._forecast_horizon = cfg.forecast_horizon
        self._days = len(dataset)/self._timeslots_per_day
        self._max_timesteps = self._days * self._timeslots_per_day - self._forecast_horizon - 1
        self._capacity = battery_cap
        self._power_battery = battery_cap/2
        self._soe = 0.0
        self._electricity_cost = 0.0
        self._batch_size = torch.Size([])
        self.observation_spec = specs[0]
        self.action_spec = specs[1]
        self._dataset = dataset
    
    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def get_max_timesteps(self):
        return int(self._max_timesteps)

    def _get_obs(self):
        return torch.cat((torch.tensor([self._soe, self._net_load], dtype=self._dtype), 
                        self._net_load_forecast, 
                        torch.tensor([self._electricity_price], dtype=self._dtype), 
                        self._electricity_price_forecast), 0)

    def _reset(self, tensordict_in):
        self._current_timestep = 0
        net_load_data, electricity_price_data = self._dataset.__getitem__(self._current_timestep)
        self._net_load = net_load_data[0]
        self._net_load_forecast = net_load_data[1:]
        self._electricity_price = electricity_price_data[0]
        self._electricity_price_forecast = electricity_price_data[1:]
        self._soe = torch.tensor([0.0], dtype=self._dtype)
        self._episode_ended =  torch.tensor(False, dtype=torch.bool)
        self._electricity_cost = 0.0
        tensordict_out = TensorDict({'observation': self._get_obs(),
                                        'done': self._episode_ended,
                                        'cost': self._electricity_cost},
                                    batch_size=self._batch_size)
        return tensordict_out

    def _step(self, tensordict_in):
        action = tensordict_in['action'].detach()
        self._current_timestep += 1

        soe_old = self._soe
        self._soe = torch.clip(soe_old + action, 0.0, self._capacity)
        penalty_soe  = torch.abs(action - (self._soe - soe_old))

        p_battery = soe_old - self._soe

        net_load_data, electricity_price_data = self._dataset.__getitem__(self._current_timestep)
        self._net_load = net_load_data[0]
        self._net_load_forecast = net_load_data[1:]
        self._electricity_price = electricity_price_data[0]
        self._electricity_price_forecast = electricity_price_data[1:]

        grid = self._net_load - p_battery
        grid_buy = grid if grid > 0.0 else 0.0
        grid_sell = torch.abs(grid) if grid <= 0.0 else 0.0

        cost = grid_buy*self._electricity_price
        profit = grid_sell*self._electricity_price
        self._electricity_cost += profit - cost

        reward = (profit - cost) - penalty_soe

        if self._current_timestep >= self._max_timesteps:
            self._episode_ended = torch.tensor(True, dtype=torch.bool)

        tensordict_out = TensorDict({'observation': self._get_obs(),
                                        'reward': reward,
                                        'done': self._episode_ended,
                                        'cost': self._electricity_cost})
        return tensordict_out