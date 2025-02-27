from dataclasses import dataclass
from typing import List

@dataclass
class Actor:
    pass

@dataclass
class Critic:
    pass

@dataclass
class NNActor(Actor):
    fc_layers: List[int]
    learning_rate: float
    ou_theta: float
    ou_sigma: float
    ou_annealing_num_steps: int

@dataclass
class NNCritic(Critic):
    obs_fc_layers: List[int]
    joint_fc_layers: List[int]
    learning_rate: float

@dataclass
class Networks:
    description: str
    actor: Actor
    critic: Critic

@dataclass
class NN(Networks):
    description: str
    actor: NNActor
    critic: NNCritic

@dataclass
class Ddpg:
    data_collector_frames_per_batch: int
    data_collector_init_frames: int
    replay_buffer_capacity: int
    train_iterations_per_frame: int
    target_update_tau: float
    td_gamma: float
    batch_size: int
    num_iterations: int
    eval_period: int

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
class HydraConfig:
    name: str
    output_path: str
    energy_dataset_path: str
    price_dataset_path: str
    mode: Mode
    env: Env
    ddpg: Ddpg
    networks: Networks
    device: str