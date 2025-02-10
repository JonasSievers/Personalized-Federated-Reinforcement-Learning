import torch
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite
from tensordict import TensorDict, TensorDictBase

from utils.EnergyDataset import EnergyDataset

class EnergyManagementEnv(EnvBase):
    def __init__(self, customer, dataset, battery_spec, cfg, device):
        super().__init__(device=device, batch_size=torch.Size([]))
        self._dtype = torch.float32
        self._customer = customer
        self._current_timestep = 0
        self._episode_ended = False
        self._timeslots_per_day = cfg.timeslots_per_day
        self._forecast_horizon = cfg.forecast_horizon
        self._days = len(dataset)/self._timeslots_per_day
        self._max_timesteps = self._days * self._timeslots_per_day - self._forecast_horizon - 1
        self._capacity = battery_spec
        self._power_battery = battery_spec/2
        self._init_charge = torch.tensor([cfg.init_charge], dtype=self._dtype)
        self._soe = torch.tensor([cfg.init_charge], dtype=self._dtype)
        self._electricity_cost = 0.0
        self._batch_size = torch.Size([])
        self._observation_shape = 2 * self._forecast_horizon + 3
        self.action_spec = Bounded(low=-self._power_battery/2, high=self._power_battery/2, shape=(1,), dtype=torch.float32)
        self.observation_spec = Composite(observation=Unbounded(shape=(self._observation_shape,), dtype=torch.float32))
        self._dataset = dataset
    
    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _get_obs(self):
        return torch.cat((torch.tensor([self._soe, self._net_load], dtype=self._dtype), 
                        self._net_load_forecast, 
                        torch.tensor([self._electricity_price], dtype=self._dtype), 
                        self._electricity_price_forecast), 0)

    def _reset(self, tensordict_in):
        self._current_timestep = 0
        self._electricity_price, self._net_load = self._dataset.getTensor(self._customer, self._current_timestep)
        self._electricity_price_forecast, self._net_load_forecast = self._dataset.getTensor(self._customer, self._current_timestep+1, self._current_timestep+self._forecast_horizon)
        self._soe = self._init_charge
        self._episode_ended =  torch.tensor(False, dtype=torch.bool)
        self._electricity_cost = 0.0
        tensordict_out = TensorDict({'observation': self._get_obs(),
                                        'done': self._episode_ended},
                                    batch_size=self._batch_size)
        return tensordict_out

    def _step(self, tensordict_in):
        action = tensordict_in['action']
        self._current_timestep += 1

        soe_old = self._soe
        self._soe = torch.clip(soe_old + action, torch.Tensor([0.0]), self._capacity).detach()
        penalty_soe  = torch.abs(action - (self._soe - soe_old)).detach()

        p_battery = soe_old - self._soe

        self._electricity_price, self._net_load = self._dataset.getTensor(customer = self._customer, idx_start = self._current_timestep)
        self._electricity_price_forecast, self._net_load_forecast = self._dataset.getTensor(self._customer, self._current_timestep+1, self._current_timestep+self._forecast_horizon)

        grid = self._net_load - p_battery
        grid_buy = grid if grid > 0.0 else 0.0
        grid_sell = torch.abs(grid) if grid <= 0.0 else 0.0

        cost = grid_buy*self._electricity_price
        profit = grid_sell*self._electricity_price
        self._electricity_cost += profit - cost

        reward = (profit - cost)

        if self._current_timestep >= self._max_timesteps:
            self._episode_ended = torch.tensor(True, dtype=torch.bool)

        tensordict_out = TensorDict({'observation': self._get_obs(),
                                        'reward': reward,
                                        'done': self._episode_ended},
                                    batch_size=self._batch_size)
        return tensordict_out