import pandas as pd
import tqdm
from utils.Networks import CustomActorNet, CustomCriticNet
import utils.utils as utils
import torch
from torchrl.modules import MLP, EGreedyModule, QValueModule, Actor,OrnsteinUhlenbeckProcessModule
from torchrl.objectives import DQNLoss, ValueEstimators, SoftUpdate, DDPGLoss
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torch.optim import Adam
from torchrl.data import LazyMemmapStorage, ReplayBuffer, RandomSampler

class Learner():
    def __init__(self, cfg: dict, device: torch.device):
        self._cfg = cfg
        self._output_path = f"{self._cfg.output_path}/{self._cfg.name}"
        self._customers = cfg.env.customer
        self._algorithm = cfg.algorithm.description
        self._device = device
        self._agents = {}
        match self._algorithm:
            case 'ddpg':
                self._setupDDPG()
            case 'dqn':
                self._setupDQN()
            
       
    def _setupDQN(self):
        for customer in self._customers:
            datasets = utils.createDatasets(cfg=self._cfg, customer=customer)
            battery_cap = utils.calcBatteryCapacity(dataset=datasets[0])
            observation_spec, action_spec = utils.createEnvSpecs(battery_cap, self._cfg)
            envs = utils.create_envs(customer, datasets, battery_cap, (observation_spec.clone(), action_spec.clone()), self._cfg, self._device)    
            value_net = MLP(in_features=observation_spec["observation"].shape[0],
                            out_features=self._cfg.algorithm.num_intervals_discretized, 
                            num_cells=self._cfg.algorithm.network.fc_layers)
            value_mod = TensorDictModule(value_net, in_keys=["observation"],out_keys=["action_value"])
            policy_module = TensorDictSequential(value_mod, QValueModule(spec=envs[0].action_spec))
            exploration_module = EGreedyModule(envs[0].action_spec)
            exploration_policy_module = TensorDictSequential(policy_module, exploration_module)
            loss_module = DQNLoss(policy_module, action_space=envs[0].action_spec)
            loss_module.make_value_estimator(ValueEstimators.TD1, gamma=self._cfg.algorithm.td_gamma)
            optimiser = Adam(loss_module.parameters(), lr=self._cfg.algorithm.network.learning_rate)

            target_updates = SoftUpdate(loss_module=loss_module, tau=self._cfg.algorithm.target_update_tau)           

            collector = SyncDataCollector(create_env_fn=(utils.create_envs(customer,
                                                                           datasets,
                                                                           battery_cap, 
                                                                           (observation_spec.clone(), action_spec.clone()), 
                                                                           self._cfg, self._device)[0]),
                                          policy=exploration_policy_module, 
                                          frames_per_batch=self._cfg.algorithm.data_collector_frames_per_batch, 
                                          total_frames=self._cfg.algorithm.num_iterations*self._cfg.algorithm.data_collector_frames_per_batch)

            buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._cfg.algorithm.replay_buffer_capacity),
                                  sampler=RandomSampler(), 
                                  batch_size=self._cfg.algorithm.batch_size)

            self._agents[customer] = {'loss': loss_module, 
                                      'target_updates': target_updates, 
                                      'optimiser': optimiser,
                                      'collector': collector,
                                      'buffer': buffer,
                                      'envs': envs}
    
    def _setupDDPG(self):
        for customer in self._customers:
            datasets = utils.createDatasets(cfg=self._cfg, customer=customer)
            battery_cap = utils.calcBatteryCapacity(dataset=datasets[0])
            observation_spec, action_spec = utils.createEnvSpecs(battery_cap, self._cfg)
            envs = utils.create_envs(customer, datasets, battery_cap, (observation_spec.clone(), action_spec.clone()), self._cfg, self._device)    

            if self._cfg.networks.description=='nn':
                actor_net = CustomActorNet(observation_spec=observation_spec.clone(),
                                           action_spec= action_spec.clone(),
                                           fc_layers=self._cfg.networks.actor.fc_layers)
                critic_net = CustomCriticNet(observation_spec=observation_spec.clone(),
                                             action_spec= action_spec.clone(),
                                             obs_fc_layers=self._cfg.networks.critic.obs_fc_layers,
                                             joint_fc_layers=self._cfg.networks.critic.joint_fc_layers)

            policy_module = Actor(module=actor_net, in_keys=["observation"], out_keys=["action"])
            critic_module = TensorDictModule(module=critic_net, in_keys=["observation", "action"], out_keys=["state_action_value"])

            loss_module = DDPGLoss(actor_network=policy_module, value_network=critic_module, delay_actor=True, delay_value=True) 
            loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self._cfg.ddpg.td_gamma)

            target_updates = SoftUpdate(loss_module=loss_module, tau=self._cfg.ddpg.target_update_tau)

            actor_optimizer = torch.optim.Adam(loss_module.actor_network.parameters(), lr=self._cfg.networks.actor.learning_rate)
            critic_optimizer = torch.optim.Adam(loss_module.value_network.parameters(), lr=self._cfg.networks.critic.learning_rate)

            ou = OrnsteinUhlenbeckProcessModule(spec=policy_module.spec.clone(),
                                                theta=self._cfg.networks.actor.ou_theta,
                                                sigma=self._cfg.networks.actor.ou_sigma,
                                                annealing_num_steps=self._cfg.ddpg.num_iterations*self._cfg.ddpg.data_collector_frames_per_batch//2)
            exploration_policy_module = TensorDictSequential(policy_module, ou)

            collector = SyncDataCollector(create_env_fn=(utils.create_envs(customer, datasets, battery_cap, (observation_spec.clone(), action_spec.clone()), self._cfg, self._device)[0]),
                                          policy=exploration_policy_module, 
                                          frames_per_batch=self._cfg.ddpg.data_collector_frames_per_batch, 
                                          total_frames=self._cfg.ddpg.num_iterations*self._cfg.ddpg.data_collector_frames_per_batch,
                                          init_random_frames=self._cfg.ddpg.data_collector_init_frames)

            buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._cfg.ddpg.replay_buffer_capacity),sampler=RandomSampler(), batch_size=self._cfg.ddpg.batch_size)

            self._agents[customer] = {'loss': loss_module, 
                                                 'target_updates': target_updates, 
                                                 'optimiser': {'loss_actor': actor_optimizer, 'loss_value': critic_optimizer},
                                                 'collector': collector,
                                                 'buffer': buffer,
                                                 'envs': envs}

    def _loadAgent(self, customer):
        return self._agents[customer]['loss'], self._agents[customer]['target_updates'], self._agents[customer]['optimiser'], self._agents[customer]['collector'], self._agents[customer]['buffer'], self._agents[customer]['envs']
    
    def train(self):
        pass