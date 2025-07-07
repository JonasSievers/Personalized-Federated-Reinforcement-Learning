from utils.Networks import CustomActorNet, CustomCriticNet
from utils.utils import make_datasets, make_env

import torch
from torch.optim import Adam
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, OrnsteinUhlenbeckProcessModule, TanhModule
from torchrl.objectives import DQNLoss, ValueEstimators, SoftUpdate, DDPGLoss
from torchrl.collectors import SyncDataCollector
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
            
       
    # def _setupDQN(self):
        # for customer in self._customers:
        #     datasets = utils.createDatasets(cfg=self._cfg, customer=customer)
        #     battery_cap = utils.calcBatteryCapacity(dataset=datasets[0])
        #     observation_spec, action_spec = utils.createEnvSpecs(self._cfg)
        #     envs = utils.create_envs(customer, datasets, battery_cap, (observation_spec.clone(), action_spec.clone()), self._cfg, self._device)    
        #     value_net = MLP(in_features=observation_spec["observation"].shape[0],
        #                     out_features=self._cfg.algorithm.num_intervals_discretized, 
        #                     num_cells=self._cfg.algorithm.network.fc_layers)
        #     value_mod = TensorDictModule(value_net, in_keys=["observation"],out_keys=["action_value"])
        #     policy_module = TensorDictSequential(value_mod, QValueModule(spec=envs[0].action_spec))
        #     exploration_module = EGreedyModule(envs[0].action_spec)
        #     exploration_policy_module = TensorDictSequential(policy_module, exploration_module)
        #     loss_module = DQNLoss(policy_module, action_space=envs[0].action_spec)
        #     loss_module.make_value_estimator(ValueEstimators.TD1, gamma=self._cfg.algorithm.td_gamma)
        #     optimiser = Adam(loss_module.parameters(), lr=self._cfg.algorithm.network.learning_rate)

        #     target_updates = SoftUpdate(loss_module=loss_module, tau=self._cfg.algorithm.target_update_tau)           

        #     collector = SyncDataCollector(create_env_fn=(utils.create_envs(customer,
        #                                                                    datasets,
        #                                                                    battery_cap, 
        #                                                                    (observation_spec.clone(), action_spec.clone()), 
        #                                                                    self._cfg, self._device)[0]),
        #                                   policy=exploration_policy_module, 
        #                                   frames_per_batch=self._cfg.algorithm.data_collector_frames_per_batch, 
        #                                   total_frames=self._cfg.algorithm.num_iterations*self._cfg.algorithm.data_collector_frames_per_batch)

        #     buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._cfg.algorithm.replay_buffer_capacity),
        #                           sampler=RandomSampler(), 
        #                           batch_size=self._cfg.algorithm.batch_size)

        #     self._agents[customer] = {'loss': loss_module, 
        #                               'target_updates': target_updates, 
        #                               'optimiser': optimiser,
        #                               'collector': collector,
        #                               'buffer': buffer,
        #                               'envs': envs}
    
    def _setupDDPG(self):
        for customer in self._customers:
            datasets = make_datasets(cfg=self._cfg, customer=customer)
            envs = make_env(cfg=self._cfg, datasets=datasets, device=self._device)
            
            policy_net = MLP(
                in_features=envs[0].observation_spec['observation'].shape[-1],
                out_features=envs[0].action_spec.shape.numel(),
                depth=2,
                num_cells=[400,300],
                activation_class=torch.nn.ReLU,
            )
          
            policy_module = TensorDictModule(
                module=policy_net,
                in_keys=['observation'],
                out_keys=['action']
            )

            actor = TensorDictSequential(
                policy_module,
                TanhModule(
                    spec=envs[0].full_action_spec['action'],
                    in_keys=['action'],
                    out_keys=['action'],
                ),
            )

            ou = OrnsteinUhlenbeckProcessModule(
                # annealing_num_steps=5_000,
                # n_steps_annealing=5_000,
                spec=actor[-1].spec.clone(),
            )

            exploration_policy = TensorDictSequential(
                actor,
                ou
            )

            critic_module = TensorDictModule(
                module=MLP(
                    in_features=envs[0].observation_spec['observation'].shape[-1] + envs[0].full_action_spec['action'].shape.numel(),
                    out_features=1,
                    depth=2,
                    num_cells=[400,300],
                    activation_class=torch.nn.ReLU,
                ),
                in_keys=['observation', 'action'],
                out_keys=['state_action_value']
            )

            collector = SyncDataCollector(create_env_fn=(make_env(cfg=self._cfg, datasets=datasets,device=self._device)[0]),
                                          policy=exploration_policy,
                                          frames_per_batch=self._cfg.algorithm.data_collector_frames_per_batch, 
                                          total_frames=self._cfg.algorithm.num_iterations*self._cfg.algorithm.data_collector_frames_per_batch)
            
            replay_buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._cfg.algorithm.replay_buffer_capacity),
                                         sampler=RandomSampler(), 
                                         batch_size=self._cfg.algorithm.batch_size)

            loss_module = DDPGLoss(actor_network=policy_module,
                                   value_network=critic_module,
                                   delay_actor=True,
                                   delay_value=True) 
            loss_module.make_value_estimator(value_type=ValueEstimators.TD0,
                                             gamma=self._cfg.algorithm.td_gamma)

            target_updater = SoftUpdate(loss_module=loss_module,
                                        tau=self._cfg.algorithm.target_update_tau)

            optimisers = {
                "loss_actor": torch.optim.Adam(params=loss_module.actor_network.parameters(), 
                                               lr=self._cfg.algorithm.network.actor_learning_rate),
                "loss_value": torch.optim.Adam(params=loss_module.value_network.parameters(), 
                                               lr=self._cfg.algorithm.network.critic_learning_rate),
            }

            self._agents[customer] = {'loss': loss_module, 
                                                 'target_updater': target_updater, 
                                                 'optimisers': optimisers,
                                                 'collector': collector,
                                                 'replay_buffer': replay_buffer,
                                                 'envs': envs}

    def _loadAgent(self, customer):
        return self._agents[customer]['loss'], self._agents[customer]['target_updater'], self._agents[customer]['optimisers'], self._agents[customer]['collector'], self._agents[customer]['replay_buffer'], self._agents[customer]['envs']
    
    def train(self):
        pass

    def test(self):
        pass    