import tqdm
import utils.utils as utils
import torch
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.modules import Actor, OrnsteinUhlenbeckProcessModule
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer, RandomSampler
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule, TensorDictSequential
from itertools import islice
from torch.utils.tensorboard import SummaryWriter



class FederatedLearner():
    def __init__(self, cfg: dict, device: torch.device):
        self._cfg = cfg
        self._customers = cfg.env.customer
        self._device = device
        self._agents = {}
        self._datasets = utils.createDatasets(self._cfg)

    def setup(self):
        self._observation_spec, self._action_spec = utils.createEnvSpecs(self._cfg)

        for customer in self._customers:
            envs = utils.create_envs(customer, self._datasets, self._cfg, self._device)

            actor_net, critic_net = utils.createActorCritic(self._cfg)
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

            collector = SyncDataCollector(create_env_fn=(utils.create_envs(customer, [self._datasets[0]], self._cfg, self._device)[0]),
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
            
    def _trainLocal(self, global_iteration, writer):
        for customer in self._customers:
            loss_module = self._agents[customer]['ddpg']
            target_updates = self._agents[customer]['target_updates']
            optimisers = self._agents[customer]['optimisers']
            collector = self._agents[customer]['collector']
            exploration_policy_module = self._agents[customer]['collector'].policy
            buffer = self._agents[customer]['buffer']

            for iteration, batch in islice(enumerate(collector), 10):
                current_frames = batch.numel()
                buffer.extend(batch)

                if global_iteration >= self._cfg.ddpg.data_collector_init_frames:
                    if global_iteration % self._cfg.ddpg.eval_period == 0:
                        eval_iteration = iteration // self._cfg.ddpg.eval_period
                        eval_env = self._agents[customer]['envs'][1]
                        obs = eval_env.reset()
                        done = torch.tensor(False)
                        while not done:
                            action = loss_module.actor_network(obs)
                            obs = eval_env.step(action)['next']
                            done = obs['done']
                        writer.add_scalar(f"{customer}/eval", eval_env.getElectricityCost(), eval_iteration)

                    for i in range(self._cfg.ddpg.train_iterations_per_frame):
                        sample = buffer.sample()
                        loss_vals = loss_module(sample)
                        for loss_name in ["loss_actor", "loss_value"]:
                            loss = loss_vals[loss_name]
                            optimiser = optimisers[loss_name]
                            writer.add_scalar(f"{customer}/{loss_name}", loss, (global_iteration+iteration)*self._cfg.ddpg.train_iterations_per_frame+i)
                            loss.backward()
                            optimiser.step()
                            optimiser.zero_grad()
                        target_updates.step()
                exploration_policy_module[-1].step(current_frames)

    def _aggregate(self):
        self._actor_params_list = []
        self._critic_params_list = []
        for customer in self._customers:
            self._actor_params_list.append(self._agents[customer]['ddpg'].actor_network_params.state_dict())
            self._critic_params_list.append(self._agents[customer]['ddpg'].value_network_params.state_dict())
    
    def _federatedAveraging(self):
        self._global_actor_params = {}
        self._global_critic_params = {}
        for key in list(self._actor_params_list[0].keys())[:-2]:
            self._global_actor_params[key] = torch.stack([params[key] for params in self._actor_params_list]).mean(dim=0)
        self._global_actor_params['__batch_size'] = self._actor_params_list[0]['__batch_size']
        self._global_actor_params['__device'] = self._actor_params_list[0]['__device']
        for key in list(self._critic_params_list[0].keys())[:-2]:
            self._global_critic_params[key] = torch.stack([params[key] for params in self._critic_params_list]).mean(dim=0)
        self._global_critic_params['__batch_size'] = self._critic_params_list[0]['__batch_size']
        self._global_critic_params['__device'] = self._critic_params_list[0]['__device']

    def _updateLocal(self):
        for customer in self._customers:
            self._agents[customer]['ddpg'].actor_network_params.load_state_dict(self._global_actor_params)
            self._agents[customer]['ddpg'].value_network_params.load_state_dict(self._global_critic_params)

    def train_eval_test(self):
        writer = SummaryWriter(log_dir=f"{self._cfg.output_path}/{self._cfg.name}/")
        
        pbar = tqdm.tqdm(total=self._cfg.ddpg.num_iterations//self._cfg.mode.num_iterations_local, desc='Federeated Learning')
        
        for i in range(self._cfg.ddpg.num_iterations//self._cfg.mode.num_iterations_local):
            pbar.set_description(f"Federated Learning Round {i+1}")
            global_iteration = i*self._cfg.mode.num_iterations_local
            self._trainLocal(global_iteration, writer)
            self._aggregate()
            self._federatedAveraging()
            self._updateLocal()
            pbar.update(1)

        for customer in self._customers:
            loss_module = self._agents[customer]['ddpg']
            test_env = self._agents[customer]['envs'][2]
            obs = test_env.reset()
            done = torch.tensor(False)
            while(done == torch.tensor(False)):
                action = loss_module.actor_network(obs)
                obs = test_env.step(action)['next']
                done = obs['done']
            writer.add_scalar(f"{customer}/cost", test_env.getElectricityCost())
        writer.close()
