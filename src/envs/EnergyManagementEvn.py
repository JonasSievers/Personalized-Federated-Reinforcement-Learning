import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EnergyManagementEnv(gym.Env):
    def __init__(self, cfg):
        self._current_timestep = 0
        self._timeslots_per_day = cfg.timeslots_per_day
        self._max_timesteps = cfg.timeslots_per_day * cfg.days
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
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(39,), dtype=np.float32)

        self.data = None
        self._p_load = None
        self._p_pv = None
        self._electricity_price = None
        self._pv_forecast = None
        self._electricity_price_forecast = None

    def _get_obs(self):
        return np.concatenate(([self._soe, self._p_load-self._p_pv, self._electricity_price], self._electricity_price_forecast, self._pv_forecast), dtype=np.float32)
    
    def _get_info(self):
        return {'Current Step':self._current_timestep,'Electricity cost': self._electricity_cost}

    def reset(self, options, seed=None):
        self._current_timestep = 0
        
        # p_load = self.data.iloc[self._current_timestep,0]
        # p_pv = self.data.iloc[self._current_timestep,1]
        # p_net_load = p_load - p_pv
        # electricity_price = self.data.iloc[self._current_timestep,2]
        # # grid_emissions = self._data.iloc[self.__current_timestep,3]

        # pv_forecast = self.data.iloc[self._current_timestep+1 : self._current_timestep+19, 1]
        # electricity_price_forecast = self.data.iloc[self._current_timestep+1 : self._current_timestep+19,2]

        self._p_load = 0
        self._p_pv = 0
        self._electricity_price = 0
        self._electricity_price_forecast = np.zeros(18)
        self._pv_forecast = np.zeros(18)

        self._soe = self._init_charge
        self._episode_ended = False
        self._electricity_cost = 0.0
        self._total_emissions = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action):
        print(self._current_timestep)
        # 1. Balance Battery
        penalty_factor = 3
        soe_old = self._soe
        self._soe = np.clip(soe_old + action[0], 0.0, self._capacity, dtype=np.float32)
        #1.1 Physikal limitations: Guides the agent to explore charging and discharging
        penalty_soe  = np.abs(action[0] - (self._soe - soe_old))*penalty_factor

        #1.2 Battery aging
        lower_threshold, upper_threshold = 0.10 * self._capacity, 0.90 * self._capacity
        if self._soe < lower_threshold:
            penalty_aging = (lower_threshold - self._soe) * penalty_factor
        elif self._soe > upper_threshold:
            penalty_aging = (self._soe - upper_threshold) * penalty_factor
        else:
            penalty_aging = 0 

        p_battery = soe_old - self._soe #Clipped to actual charging. + -> discharging/ providing energy
        
        #2. Get data
        # p_load = self._data.iloc[self.__current_timestep, 0] 
        p_load = 0
        # p_pv = self._data.iloc[self.__current_timestep, 1] 
        p_pv = 0
        p_net_load = p_load - p_pv
        p_net_load = 0
        # electricity_price = self._data.iloc[self.__current_timestep, 2]
        electricity_price = 0
        # grid_emissions = self._data.iloc[self.__current_timestep, 3]
        grid_emissions = 0

        #2.1 Get forecasts
        # pv_forecast = self._data.iloc[self.__current_timestep+1 : self.__current_timestep+19, 1]
        pv_forecast = 0
        # price_forecast = self._data.iloc[self.__current_timestep+1 : self.__current_timestep+19, 2]
        price_forecast = 0
        
        #3. Balance Grid
        grid = p_load - p_pv - p_battery
        grid_buy = grid if grid > 0 else 0
        grid_sell = abs(grid) if grid < 0 else 0

        #4. Calculate profit
        cost = grid_buy*electricity_price
        profit = grid_sell*self._feed_in_price
        self._electricity_cost += profit - cost

        #4.1 Calculate emissions
        emissions = grid_buy*grid_emissions
        self._total_emissions += emissions

        emissions_penalty_factor = 0.05  # This value could be adjusted based on how severely you want to penalize emissions
        emissions_impact = emissions * emissions_penalty_factor

        reward_scaling_factor = 5
        reward = ((profit - cost)*reward_scaling_factor)*(1-self._ecoPriority) - (self._ecoPriority * emissions_impact) - penalty_soe - penalty_aging

        #6. Create observation
        observation = self._get_obs()
        info = self._get_info()
        
  
        # # Logging
        # if self._logging:
        #     wandb.log({
        #     'Action [2.3, -2.3]': action[0], 
        #     'SoE [0, 13.5]': self._soe, 
        #     'Battery wear cost': penalty_aging,
        #     'Profit (+ profit, - cost)': profit - cost,
        #     'Reward' : reward,
        #     'PV': p_pv, 
        #     'Load' : p_load, 
        #     'Price' : electricity_price,
        #     'Net load': p_net_load,
        #     'Emissions [kg]': emissions,
        #     })

        # Check for episode end
        # if self.__current_timestep >= self._max_timesteps - 19:
        if self._current_timestep >= 2:
            self._episode_ended = True
        else:
            self._current_timestep += 1
            # if self._logging:
            #     wandb.log({'Final Profit': self._electricity_cost})
            #     wandb.log({'Final Emissions': self._total_emissions}) 
        truncated = False          
        return observation, reward, self._episode_ended, truncated, info