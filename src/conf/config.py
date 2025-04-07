from dataclasses import dataclass
from typing import List

@dataclass
class Networks:
    description: str

@dataclass
class DDPG_NN(Networks):
    description: str
    actor_fc_layers: List[int]
    actor_learning_rate: float
    critic_obs_fc_layers: List[int]
    critic_joint_fc_layers: List[int]
    critic_learning_rate: float
@dataclass
class DQN_NN(Networks):
    description: str
    fc_layers: List[int]
    learning_rate: float
    
@dataclass
class Env:
    customer: List[int]
    timeslots_per_day: int
    forecast_horizon: int
    init_charge: float

@dataclass
class Mode:
    description: str

@dataclass
class Local(Mode):
    description: str

@dataclass
class Fed(Mode):
    description: str
    num_fed_rounds: int

@dataclass
class Per(Mode):
    description: str
    num_fed_rounds: int
    num_shared_layers: int
@dataclass
class Algorithm:
    description: str

@dataclass
class DDPG(Algorithm):
    description: str
    num_iterations: int
    eval_period: int
    batch_size: int
    td_gamma: float
    target_update_tau: float
    ou_theta: float
    ou_sigma: float
    ou_annealing_num_steps: int
    network: Networks
    data_collector_frames_per_batch: int
    replay_buffer_capacity: int
    train_iterations_per_frame: int

@dataclass
class DQN(Algorithm):
    description: str
    num_iterations: int
    eval_period: int
    batch_size: int
    td_gamma: float
    target_update_tau: float
    network: Networks
    num_intervals_discretized: int
    data_collector_frames_per_batch: int
    replay_buffer_capacity: int
    train_iterations_per_frame: int
   
@dataclass
class HydraConfig:
    name: str
    output_path: str
    model_path: str
    energy_dataset_path: str
    price_dataset_path: str
    use_forecast: bool
    mode: Mode
    env: Env
    algorithm: Algorithm
    networks: Networks
    device: str