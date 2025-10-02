import torch
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite
from tensordict import TensorDict, TensorDictBase

class BatteryScheduling(EnvBase):
    def __init__(self, cfg, datasets, device):
        super().__init__()

        # Cfg
        self.cfg = cfg
        self.DEVICE = device

        # Dataset
        self.datasets = datasets
        self.ds_number = None

        # Environment parameters
        td = self._make_params()

        # Environment specs
        self._make_specs(td)

        # Environment seed
        self.set_seed(cfg.seed)


    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng


    def _reset(self, td_in):
        if td_in is None or (len(td_in.keys())==1):
           td_in = self._make_params()
        else:
            td_in['params'] = self._make_params()['params']

        # Use random dataset while training and always first day during eval/test
        if self.training:
            self.ds_number = torch.randint(low=0,high=len(self.datasets)-1,size=(), device=self.DEVICE)
            init_soe = torch.rand((), device=self.DEVICE)* td_in['params','battery_capacity']
        else:
            if self.ds_number is None:
                self.ds_number = 0
            else:
                self.ds_number +=1
            init_soe = torch.tensor(0.0, device=self.DEVICE)
            
        self._current_dataset = self.datasets[self.ds_number]

        step = torch.tensor(0, dtype=torch.int64, device=self.DEVICE)
        prosumption = self._current_dataset['prosumption'][step]
        prosumption_forecast = self._current_dataset['prosumption'][step:step+self.cfg.dataset.forecast_horizon-1]
        price = self._current_dataset['price'][step]
        price_forecast = self._current_dataset['price'][step:step+self.cfg.dataset.forecast_horizon-1]

        td_out = TensorDict(
            {
                'step': step,
                'soe': init_soe,
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': torch.tensor(0.0, device=self.DEVICE),
                'params': td_in['params'],
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )
        return td_out

    def _step(self, td_in):
        action = td_in['action'].squeeze(-1).detach()
        step = td_in['step'] + 1
        old_soe = td_in['soe']
        params = td_in['params']

        prosumption = self._current_dataset['prosumption'][step]
        prosumption_forecast = self._current_dataset['prosumption'][step:step+self.cfg.dataset.forecast_horizon-1]
        price = self._current_dataset['price'][step]
        price_forecast = self._current_dataset['price'][step:step+self.cfg.dataset.forecast_horizon-1]

        new_soe = torch.clip(old_soe + action, torch.tensor(0.0, device=self.DEVICE), params['battery_capacity'])
        clipped_action = new_soe - old_soe
        penalty_soe  = torch.abs(action - clipped_action)

        grid = prosumption + clipped_action
        
        cost =  grid*price if grid>= 0 else grid*0.1
        reward = -cost - penalty_soe
        
        td_out = TensorDict(
            {
                'step': step,
                'soe': new_soe,
                'prosumption': prosumption,
                'prosumption_forecast': prosumption_forecast,
                'price': price,
                'price_forecast': price_forecast,
                'cost': cost,
                'params': params,
                'reward': reward,
                'done': ((step + 1) > params['max_steps']),
            },
            batch_size=td_in.shape,
            device=td_in.device,
        )
        
        return td_out
    

    def _make_params(self):
        battery_cap = self.datasets.getBatteryCapacity()
        max_steps = self.datasets[0].shape[0]-self.cfg.dataset.forecast_horizon
        td_param = TensorDict(
            {
                'params': TensorDict(
                    {
                        'battery_capacity': battery_cap,
                        'max_power': battery_cap/4,
                        'max_steps': max_steps
                    },
                    batch_size=torch.Size([])
                )
            },
            batch_size=torch.Size([]),
            device=self.DEVICE,
        )
        return td_param
    
    def _make_specs(self, td_param):
        self.observation_spec = Composite(
            step=Bounded(low=0,
                         high= td_param['params', 'max_steps'],
                         shape=(),
                         dtype=torch.int64),
            soe=Bounded(low = 0,
                         high = td_param['params', 'battery_capacity'],
                         shape=(),
                         dtype=torch.float32),
            prosumption=Unbounded(dtype=torch.float32, 
                                  shape=()),
            prosumption_forecast=Unbounded(dtype=torch.float32, 
                                            shape=(self.cfg.dataset.forecast_horizon-1,)),
            price=Unbounded(dtype=torch.float32,
                            shape=()),
            price_forecast=Unbounded(dtype=torch.float32, 
                                     shape=(self.cfg.dataset.forecast_horizon-1,)),
            cost=Unbounded(dtype=torch.float32, 
                           shape=()),
            params=self._make_componentosite_from_td(td_param['params']),
            shape=torch.Size([])
        )

        self.action_spec = Bounded(
            low=-td_param['params', 'max_power']/2,
            high=td_param['params', 'max_power']/2,
            shape=(1,),
            dtype=torch.float32,
        )
        
        self.reward_spec = Unbounded(shape=(*td_param.shape, 1), dtype=torch.float32)
    
    def _make_componentosite_from_td(self, td):
        componentosite = Composite(
            {
                key: self.make_componentosite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
                for key, tensor in td.items()
            },
            shape=td.shape
        )
        return componentosite