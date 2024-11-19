import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EnergyManagementEnv(gym.Env):
    def __init__(self, customer, dataset, cfg):
        self._customer = customer
        self._current_timestep = 0
        self._timeslots_per_day = cfg.timeslots_per_day
        self._forecast_horizon = cfg.forecast_horizon
        self._max_timesteps = cfg.timeslots_per_day * cfg.days - cfg.forecast_horizon - 1
        self._capacity = cfg.capacity 
        self._power_battery = cfg.power_battery
        self._init_charge = cfg.init_charge
        self._soe = cfg.init_charge
        self._episode_ended = False
        self._electricity_cost = 0.0
        self._total_emissions = 0.0
        self._logging = cfg.logging
        self._feed_in_price = cfg.feed_in_price
        self._ecoPriority = cfg.ecoPriority

        self.action_space = spaces.Box(low=-self._power_battery/2, high=self._power_battery/2, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(27,), dtype=np.float32)

        self._dataset = dataset

    def _get_obs(self):
        return np.concatenate(([self._soe, self._p_load-self._p_pv, self._electricity_price], self._electricity_price_forecast, self._pv_forecast), dtype=np.float32)
    
    def _get_info(self):
        return {'Current Step':self._current_timestep,'Electricity cost': self._electricity_cost}

    def reset(self, options, seed=None):
        self._current_timestep = 0
        self._electricity_price, _, self._p_load, self._p_pv= self._dataset.getTensor(self._customer, self._current_timestep)
        self._electricity_price_forecast, _, _, self._pv_forecast = self._dataset.getTensor(self._customer, self._current_timestep+1, self._current_timestep+self._forecast_horizon)

        self._soe = self._init_charge
        self._episode_ended = False
        self._electricity_cost = 0.0
        self._total_emissions = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._current_timestep += 1

        penalty_factor = 3
        soe_old = self._soe
        self._soe = np.clip(soe_old + action[0], 0.0, self._capacity, dtype=np.float32)
        penalty_soe  = np.abs(action[0] - (self._soe - soe_old))*penalty_factor

        lower_threshold, upper_threshold = 0.10 * self._capacity, 0.90 * self._capacity
        if self._soe < lower_threshold:
            penalty_aging = (lower_threshold - self._soe) * penalty_factor
        elif self._soe > upper_threshold:
            penalty_aging = (self._soe - upper_threshold) * penalty_factor
        else:
            penalty_aging = 0 

        p_battery = soe_old - self._soe

        self._electricity_price, _, self._p_load, self._p_pv= self._dataset.getTensor(self._customer, self._current_timestep)
        self._electricity_price_forecast, _, _, self._pv_forecast = self._dataset.getTensor(self._customer, self._current_timestep+1, self._current_timestep+self._forecast_horizon)

        grid = self._p_load - self._p_pv - p_battery
        grid_buy = grid if grid > 0 else 0
        grid_sell = abs(grid) if grid < 0 else 0

        cost = grid_buy*self._electricity_price
        profit = grid_sell*self._feed_in_price
        self._electricity_cost += profit - cost

        reward_scaling_factor = 5
        reward = (((profit - cost)*reward_scaling_factor) - penalty_soe - penalty_aging)
        reward = reward.cpu().item()

        observation = self._get_obs()
        info = self._get_info()

        if self._current_timestep >= self._max_timesteps:
            print(self._current_timestep)
            self._episode_ended = True
            
        truncated = False          
        return observation, reward, self._episode_ended, truncated, info