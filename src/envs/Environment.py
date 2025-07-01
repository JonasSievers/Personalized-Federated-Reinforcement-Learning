import torch
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite
from tensordict import TensorDict, TensorDictBase

from utils.EnergyDataset import EnergyDataset

class BatteryScheduling(EnvBase):
    def __init__(self, cfg, dataset, device):
        super().__init__(device=cfg.device, batch_size=torch.Size([]))
        
        # Cfg
        self._cfg = cfg

        # Dataset
        self._dataset = dataset

        # Environment parameters
        td = self._make_params()
        self._data_pointer = torch.tensor(0, dtype=torch.int64)

        # Environment specs
        self._make_specs(td)

        # Environment seed
        self.set_seed(cfg.seed)
    
    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def _reset(self, td_in):
        if td_in is None or td_in.is_empty():
           td_in = self._make_params()
        else:
            td_in['params'] = self._make_params()['params']

        step = torch.tensor(0, dtype=torch.int64)
        prosumption_data, price_data, time_data = self._dataset[self._data_pointer]
        prosumption = prosumption_data[0].item()
        prosumption_forecast = prosumption_data[1:]
        price = price_data[0].item()
        price_forecast = price_data[1:]
        time_feature = torch.tensor([time_data[0][0], time_data[0][1]])

        if self._data_pointer + 1 >= len(self._dataset):
            self._data_pointer = torch.tensor(0, dtype=torch.int64)


        td_out = TensorDict(
            {
                'step': step,
                'data_pointer': self._data_pointer,
                'time_feature': time_feature,
                'soe': torch.tensor(0.0),
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': torch.tensor(0.0),
                'params': td_in['params'],
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )
        return td_out

    def _step(self, td_in):
        action = td_in['action'].squeeze().detach()
        step = td_in['step'] + 1
        old_soe = td_in['soe']
        old_cost = td_in['cost']
        params = td_in['params']

        self._data_pointer += 1
        prosumption_data, price_data, time_data = self._dataset[self._data_pointer]
        prosumption = prosumption_data[0]
        prosumption_forecast = prosumption_data[1:]
        price = price_data[0]
        price_forecast = price_data[1:]
        time_feature = torch.tensor([time_data[0][0], time_data[0][1]])

        new_soe = torch.clip(old_soe + action, torch.tensor(0.0), params['battery_capacity'])
        clipped_action = new_soe - old_soe
        penalty_soe  = torch.abs(action - clipped_action)

        grid = prosumption + clipped_action
        
        cost =  grid*price if grid>= 0 else grid*-0.1
        new_cost = old_cost + cost
        reward = -cost - penalty_soe

        
        td_out = TensorDict(
            {
                'step': step,
                'data_pointer': self._data_pointer,
                'time_feature': time_feature,
                'soe': new_soe,
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': new_cost,
                'params': params,
                'reward': reward,
                'done': ((step + 1) >= 48) or (self._data_pointer + 1 >= len(self._dataset)),
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )
        print(self._data_pointer)
        print(td_out['done'])
        return td_out
    
    def _make_params(self):
        td_param = TensorDict(
            {
                'params': TensorDict(
                    {
                        'battery_capacity': self._dataset.getBatteryCapacity(),
                        'max_power': self._dataset.getBatteryCapacity()/2,
                        'max_steps': len(self._dataset)
                    },
                    batch_size=torch.Size([])
                )
            },
            batch_size=torch.Size([]),
        )
        return td_param
    
    def _make_specs(self, td_param):
        self.observation_spec = Composite(
            step=Bounded(low=0,
                         high=48,
                         shape=(),
                         dtype=torch.int64),
            time_feature=Bounded(low=0,
                            high=1,
                            shape=(2,),
                            dtype=torch.float32),
            soe=Bounded(low = 0,
                         high = td_param['params', 'battery_capacity'],
                         shape=(),
                         dtype=torch.float32),
            prosumption=Unbounded(dtype=torch.float32, 
                                  shape=()),
            prosumption_forecast=Unbounded(dtype=torch.float32, 
                                            shape=(self._cfg.env.forecast_horizon,)),
            price=Unbounded(dtype=torch.float32,
                            shape=()),
            price_forecast=Unbounded(dtype=torch.float32, 
                                     shape=(self._cfg.env.forecast_horizon,)),
            cost=Unbounded(dtype=torch.float32, 
                           shape=()),
            params=self._make_composite_from_td(td_param['params']),
            shape=torch.Size([]),
        )

        self.action_spec = Bounded(
            low=-td_param['params', 'max_power']/4,
            high=td_param['params', 'max_power']/4,
            shape=torch.Size([]),
            dtype=torch.float32,
        )
        self.reward_spec = Unbounded(shape=(*td_param.shape, 1), dtype=torch.float32)
    
    def _make_composite_from_td(self, td):
        composite = Composite(
            {
                key: self.make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
                for key, tensor in td.items()
            },
            shape=td.shape,
        )
        return composite