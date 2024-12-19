from typing import OrderedDict
from utils.Networks import CustomActorNet, CustomCriticNet
import utils.utils as utils
import torch
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.modules import Actor, OrnsteinUhlenbeckProcessModule
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer, RandomSampler
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict
from itertools import islice
from torch.utils.tensorboard import SummaryWriter



class PersonalizedLearner():
    def __init__(self, customers: list, cfg: dict):
        self._customers = customers
        self._cfg = cfg
        self._agents = {}

    def setup(self):
        self._observation_spec, self._action_spec = utils.createEnvSpecs(self._cfg.experiment.env_params)

        for customer in self._customers:
            datasets = utils.createDatasets(self._cfg)

            envs = utils.create_envs(customer, datasets, self._cfg)

            actor_net = CustomActorNet(observation_spec=self._observation_spec.clone(),
                                       action_spec=self._action_spec.clone(),
                                       fc_layers=self._cfg.experiment.params.actor.fc_layers)
            policy_module = Actor(module=actor_net, in_keys=["observation"], out_keys=["action"])

            critic_net =CustomCriticNet(observation_spec=self._observation_spec.clone(),
                                        action_spec=self._action_spec.clone(),
                                        obs_fc_layers=self._cfg.experiment.params.critic.obs_fc_layers,
                                        joint_fc_layers=self._cfg.experiment.params.critic.joint_fc_layers)
            critic_module = TensorDictModule(module=critic_net, in_keys=["observation", "action"], out_keys=["state_action_value"])

            loss_module = DDPGLoss(actor_network=policy_module, value_network=critic_module, delay_actor=True, delay_value=True)
            loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self._cfg.experiment.params.td_gamma)

            target_updates = SoftUpdate(loss_module=loss_module, tau=self._cfg.experiment.params.target_update_tau)

            actor_optimizer = torch.optim.Adam(loss_module.actor_network.parameters(), lr=self._cfg.experiment.params.actor.learning_rate)
            critic_optimizer = torch.optim.Adam(loss_module.value_network.parameters(), lr=self._cfg.experiment.params.critic.learning_rate)

            ou = OrnsteinUhlenbeckProcessModule(spec=policy_module.spec.clone(), theta=0.15, sigma=0.2, annealing_num_steps=1000)
            exploration_policy_module = TensorDictSequential(policy_module, ou)

            collector = SyncDataCollector(create_env_fn=(utils.create_envs(customer, [datasets[0]], self._cfg)[0]),
                                          policy=exploration_policy_module, 
                                          frames_per_batch=self._cfg.experiment.params.data_collector_frames_per_batch, 
                                          total_frames=self._cfg.experiment.params.num_iterations*self._cfg.experiment.params.data_collector_frames_per_batch)

            buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._cfg.experiment.params.replay_buffer_capacity),sampler=RandomSampler())


            self._agents[customer] = {'ddpg': loss_module, 
                                                 'target_updates': target_updates, 
                                                 'optimisers': {'loss_actor': actor_optimizer, 'loss_value': critic_optimizer},
                                                 'collector': collector,
                                                 'buffer': buffer,
                                                 'datasets': datasets,
                                                 'envs': envs}
            

    def trainLocal(self, num_iterations: int):
        writer = SummaryWriter(log_dir=f"{self._cfg.experiment.path}/{self._cfg.experiment.name}")
        for customer in self._customers:
            print(customer)
            loss_module = self._agents[customer]['ddpg']
            target_updates = self._agents[customer]['target_updates']
            optimisers = self._agents[customer]['optimisers']
            collector = self._agents[customer]['collector']
            buffer = self._agents[customer]['buffer']

            for iteration, batch in islice(enumerate(collector), num_iterations):
                buffer.extend(batch)
                sample = buffer.sample(self._cfg.experiment.params.batch_size)

                loss_vals = loss_module(sample)

                for loss_name in ["loss_actor", "loss_value"]:
                    loss = loss_vals[loss_name]
                    optimiser = optimisers[loss_name]
                    writer.add_scalar(f"{customer}/{loss_name}", loss, iteration)
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                if iteration % 5 == 0:
                    target_updates.step()
        writer.close()

    def aggregate(self):
        self._actor_params_list = []
        self._critic_params_list = []
        for customer in self._customers:
            self._actor_params_list.append(self._agents[customer]['ddpg'].actor_network_params.state_dict())
            self._critic_params_list.append(self._agents[customer]['ddpg'].value_network_params.state_dict())
    
    def federatedAveraging(self):
        self._global_actor_params = {}
        self._global_critic_params = {}
        # TODO adapt index to deeper layers
        for key in list(self._actor_params_list[0].keys())[:-2]:
            self._global_actor_params[key] = torch.stack([params[key] for params in self._actor_params_list]).mean(dim=0)
        self._global_actor_params['__batch_size'] = self._actor_params_list[0]['__batch_size']
        self._global_actor_params['__device'] = self._actor_params_list[0]['__device']
        for key in list(self._critic_params_list[0].keys())[:-2]:
            self._global_critic_params[key] = torch.stack([params[key] for params in self._critic_params_list]).mean(dim=0)
        self._global_critic_params['__batch_size'] = self._critic_params_list[0]['__batch_size']
        self._global_critic_params['__device'] = self._critic_params_list[0]['__device']

    def updateLocal(self):
        for customer in self._customers:
            self._agents[customer]['ddpg'].actor_network_params.load_state_dict(self._global_actor_params)
            self._agents[customer]['ddpg'].value_network_params.load_state_dict(self._global_critic_params)
