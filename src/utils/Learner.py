import os
from utils.Networks import CustomActorNet, CustomCriticNet
import utils.utils as utils
import torch
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.modules import Actor, OrnsteinUhlenbeckProcessModule
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer, RandomSampler
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule, TensorDictSequential

class Learner():
    def __init__(self, cfg: dict, device: torch.device):
        self._cfg = cfg
        self._output_path = f"{self._cfg.output_path}/{self._cfg.name}"
        self._customers = cfg.env.customer
        self._device = device
        self._agents = {}
        self._datasets = utils.createDatasets(self._cfg)
        self._battery_spec = utils.calcBatteryCapacity(self._customers, self._datasets[0])
       
    def setup(self):
        for customer in self._customers:
            self._observation_spec, self._action_spec = utils.createEnvSpecs(self._battery_spec[customer], self._cfg)
            envs = utils.create_envs(customer, self._datasets, self._battery_spec[customer], self._cfg, self._device)

            if self._cfg.networks.description=='nn':
                actor_net = CustomActorNet(observation_spec=self._observation_spec.clone(),
                                            action_spec= self._action_spec.clone(),
                                            fc_layers=self._cfg.networks.actor.fc_layers)
                critic_net = CustomCriticNet(observation_spec=self._observation_spec.clone(),
                                                action_spec= self._action_spec.clone(),
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

            collector = SyncDataCollector(create_env_fn=(utils.create_envs(customer, [self._datasets[0]], self._battery_spec[customer], self._cfg, self._device)[0]),
                                          policy=exploration_policy_module, 
                                          frames_per_batch=self._cfg.ddpg.data_collector_frames_per_batch, 
                                          total_frames=self._cfg.ddpg.num_iterations*self._cfg.ddpg.data_collector_frames_per_batch,
                                          init_random_frames=self._cfg.ddpg.data_collector_init_frames)

            buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._cfg.ddpg.replay_buffer_capacity),sampler=RandomSampler(), batch_size=self._cfg.ddpg.batch_size)

            self._agents[customer] = {'ddpg': loss_module, 
                                                 'target_updates': target_updates, 
                                                 'optimisers': {'loss_actor': actor_optimizer, 'loss_value': critic_optimizer},
                                                 'collector': collector,
                                                 'buffer': buffer,
                                                 'envs': envs}

    def train_eval_test(self):
        pass

    def loadAgent(self, customer):
        return self._agents[customer]['ddpg'], self._agents[customer]['target_updates'], self._agents[customer]['optimisers'], self._agents[customer]['collector'], self._agents[customer]['buffer'], self._agents[customer]['envs']

    def save_results(self, df, customer):
        os.makedirs(self._output_path, exist_ok=True)
        filename = f"{self._output_path}/building_{customer}_results.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")