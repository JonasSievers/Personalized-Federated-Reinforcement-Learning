import torch
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite
from tensordict import TensorDict, TensorDictBase

class EnergyManagementEnv(EnvBase):
    def __init__(self, customer, dataset, cfg, test_days=None, device='cpu'):
        self._dtype = torch.float32
        self._customer = customer
        self._current_timestep = 0
        self._timeslots_per_day = cfg.timeslots_per_day
        self._forecast_horizon = cfg.forecast_horizon
        if test_days is None:
            self._max_timesteps = cfg.timeslots_per_day * cfg.days - cfg.forecast_horizon - 1
        else:
            self._max_timesteps = cfg.timeslots_per_day * test_days - cfg.forecast_horizon - 1
        self._capacity = torch.tensor([cfg.capacity], dtype=self._dtype)
        self._power_battery = cfg.power_battery
        self._init_charge = cfg.init_charge
        self._soe = torch.tensor([cfg.init_charge], dtype=self._dtype)
        self._episode_ended = False
        self._electricity_cost = 0.0
        self._batch_size = torch.Size([])

        super().__init__(device=device, batch_size=self._batch_size)

        self.action_spec = Bounded(low=-self._power_battery/2, high=self._power_battery/2, shape=(1,), dtype=torch.float32)
        self.observation_spec = Composite(observation=Unbounded(shape=(27,), dtype=torch.float32))

        self._dataset = dataset
    
    def printPrice(self):
        return self._electricity_cost
    
    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _get_obs(self):
        return torch.cat((torch.tensor([self._soe, self._p_load-self._p_pv, self._electricity_price],dtype=self._dtype), self._electricity_price_forecast, self._pv_forecast), 0)


    def _reset(self, tensordict_in):
        self._current_timestep = 0
        self._electricity_price, _, self._p_load, self._p_pv = self._dataset.getTensor(self._customer, self._current_timestep)
        self._electricity_price_forecast, _, _, self._pv_forecast = self._dataset.getTensor(self._customer, self._current_timestep+1, self._current_timestep+self._forecast_horizon)
        self._soe = self._init_charge
        self._episode_ended =  torch.tensor(False, dtype=torch.bool)
        self._electricity_cost = 0.0
        tensordict_out = TensorDict({'observation': self._get_obs()},
                         batch_size=self._batch_size)
        return tensordict_out

    def _step(self, tensordict_in):
        action = tensordict_in['action']
        self._current_timestep += 1

        penalty_factor = 3
        soe_old = self._soe
        self._soe = torch.clip(soe_old + action, torch.Tensor([0.0]), self._capacity).detach()
        penalty_soe  = torch.abs(action - (self._soe - soe_old))*penalty_factor

        lower_threshold, upper_threshold = 0.10 * self._capacity, 0.90 * self._capacity
        if self._soe < lower_threshold:
            penalty_aging = (lower_threshold - self._soe) * penalty_factor
        elif self._soe > upper_threshold:
            penalty_aging = (self._soe - upper_threshold) * penalty_factor
        else:
            penalty_aging = 0 

        p_battery = soe_old - self._soe

        self._electricity_price, _, p_load, p_pv = self._dataset.getTensor(self._customer, self._current_timestep)
        self._electricity_price_forecast, _, _, self._pv_forecast = self._dataset.getTensor(self._customer, self._current_timestep+1, self._current_timestep+self._forecast_horizon)

        grid = p_load - p_pv - p_battery
        grid_buy = grid if grid > 0.0 else 0.0
        grid_sell = torch.abs(grid) if grid < 0.0 else 0.0

        cost = grid_buy*self._electricity_price
        profit = grid_sell*self._feed_in_price
        self._electricity_cost += profit - cost

        reward_scaling_factor = 10
        reward = ((profit - cost)*reward_scaling_factor) - penalty_soe - penalty_aging


        if self._current_timestep >= self._max_timesteps:
            self._episode_ended = torch.tensor(True, dtype=torch.bool)

        tensordict_out = TensorDict({'observation': self._get_obs(),
                                        'reward': reward,
                                        'done': self._episode_ended},
                                    batch_size=self._batch_size)
        return tensordict_out