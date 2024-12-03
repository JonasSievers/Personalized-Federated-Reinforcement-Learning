from dataclasses import dataclass
from typing import List


@dataclass
class Federated:
    num_train_iterations: int

@dataclass
class Env_Params:
    customer: List[int]
    timeslots_per_day: int
    forecast_horizon: int
    capacity: float
    power_battery: float
    init_charge: float

@dataclass
class Actor:
    fc_layers: List[int]
    learning_rate: float
    ou_theta: float
    ou_sigma: float
    ou_annealing_num_steps: int

@dataclass
class Critic:
    obs_fc_layers: List[int]
    joint_fc_layers: List[int]
    learning_rate: float

@dataclass
class Params:
    actor: Actor
    critic: Critic
    num_iterations: int
    data_collector_frames_per_batch: int
    replay_buffer_capacity: int
    target_update_period: int
    target_update_tau: float
    td_gamma: float
    batch_size: int
    eval_period: int

@dataclass
class Experiment:
    path: str
    name: str
    dataset_path: str
    federated: Federated
    env_params: Env_Params
    params: Params

@dataclass
class HydraConfig:
    experiment: Experiment