from dataclasses import dataclass

@dataclass
class Datasets:
    path: str

@dataclass
class Env_Params:
    init_charge: float
    timeslots_per_day: int
    forecast_horizon: int
    days: int
    capacity: float
    power_battery: float
    ecoPriority: float
    logging: bool
    feed_in_price: float

@dataclass
class Params:
    num_iterations: int
    initial_collect_steps: int
    collect_steps_per_iteration: int
    replay_buffer_capacity: int 
    ou_stddev: float
    ou_damping: float
    target_update_tau: float
    target_update_period: int
    batch_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    gamma: float
    reward_scale_factor: float

@dataclass
class HydraConfig:
    datasets: Datasets
    env_params: Env_Params
    params: Params