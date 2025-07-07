import torch
from torchrl.envs import Transform
from tensordict import TensorDict, TensorDictBase


"""
    Running mean and std estimator for online normalization.
"""
class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean = torch.tensor(0.0)
        self.var = torch.tensor(1.0)
        self.count = torch.tensor(0.0)

    def update(self, x):
        obs_mean = x.mean()
        obs_var = x.var(unbiased=False)

        delta = obs_mean - self.mean
        total_count = self.count + 1

        new_mean = self.mean + delta / total_count
        m_a = self.var * self.count
        m_b = obs_var
        M2 = m_a + m_b + delta**2 * self.count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        std = torch.sqrt(self.var + 1e-8)
        return (x - self.mean) / std

'''
A custom observation standardization transform.
'''
class CustomObservationStandardization(Transform):
    rms_prosumption = RunningMeanStd()
    rms_price = RunningMeanStd()
    
    def __init__(self, battery_cap):
        super().__init__(in_keys=['soe', 'prosumption', 'prosumption_forecast', 'price', 'price_forecast'], 
                         out_keys=['soe', 'prosumption', 'prosumption_forecast', 'price', 'price_forecast'], 
                         in_keys_inv=[], out_keys_inv=[])
        self.battery_cap = battery_cap
        self._eval_mode = False
        
    def _call(self, td_in):
        soe = td_in['soe']
        prosumption = td_in['prosumption']
        prosumption_forecast = td_in['prosumption_forecast']
        price = td_in['price']
        price_forecast = td_in['price_forecast']

        if not self._eval_mode:
            self.rms_prosumption.update(prosumption)
            self.rms_price.update(price)
        
        soe_norm = soe/self.battery_cap
        prosumption_norm = self.rms_prosumption.normalize(prosumption)
        prosumption_forecast_norm = self.rms_prosumption.normalize(prosumption_forecast)
        price_norm = self.rms_price.normalize(price)
        price_forecast_norm = self.rms_price.normalize(price_forecast)

        td_out = TensorDict({
            'soe': soe_norm,
            'prosumption': prosumption_norm,
            'prosumption_forecast': price_forecast_norm,
            'price': price_norm,
            'price_forecast': prosumption_forecast_norm
        })
    
        return td_in.update(td_out)
    
    def _reset(self, tensordict, tensordict_reset):
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset
    
    def eval(self):
        self._eval_mode = True

    def train(self, mode: bool = True):
        self._eval_mode = False


'''
A custom action scaler transform.
'''
class CustomActionScaler(Transform):
    def __init__(self, max_action):
        super().__init__(in_keys=[], out_keys=[], in_keys_inv=["action"], out_keys_inv=["action"])
        self._max_action = max_action

    def _inv_apply_transform(self, action):
        scaled_action = -self._max_action + self._max_action * (action + 1)
        return scaled_action
