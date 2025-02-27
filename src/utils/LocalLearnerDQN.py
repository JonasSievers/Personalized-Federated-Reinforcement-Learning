import pandas as pd
from utils.Learner import Learner
import utils.utils as utils
import torch
from torchrl.modules import MLP, QValueActor, EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, ValueEstimators, SoftUpdate
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torch.optim import Adam
from torchrl.data import LazyMemmapStorage, ReplayBuffer, RandomSampler, Categorical
import tqdm

class LocalLearnerDQN(Learner):
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)

    def setup(self):
       for customer in self._customers:
            datasets = utils.createDatasets(cfg=self._cfg, customer=customer)
            battery_cap = utils.calcBatteryCapacity(dataset=datasets[0])
            observation_spec, action_spec = utils.createEnvSpecs(battery_cap, self._cfg)
            envs = utils.create_envs(customer, datasets, battery_cap, (observation_spec.clone(), action_spec.clone()), self._cfg, self._device)       
            value_net = MLP(out_features=50, num_cells=[400,300])
            value_mod = TensorDictModule(value_net, in_keys=["observation"],out_keys=["action_value"])
            policy_module = TensorDictSequential(value_mod, QValueModule(spec=envs[0].action_spec))
            exploration_module = EGreedyModule(
                envs[0].action_spec, annealing_num_steps=100_000, eps_init=0.5
            )
            exploration_policy_module = TensorDictSequential(policy_module, exploration_module)
            loss_module = DQNLoss(policy_module, action_space=envs[0].action_spec)
            loss_module.make_value_estimator(ValueEstimators.TD1, gamma=self._cfg.ddpg.td_gamma)
            optimiser = Adam(loss_module.parameters(), lr=self._cfg.networks.actor.learning_rate)

            target_updates = SoftUpdate(loss_module=loss_module, tau=self._cfg.ddpg.target_update_tau)           

            collector = SyncDataCollector(create_env_fn=(utils.create_envs(customer, datasets, battery_cap, (observation_spec.clone(), action_spec.clone()), self._cfg, self._device)[0]),
                                          policy=exploration_policy_module, 
                                          frames_per_batch=self._cfg.ddpg.data_collector_frames_per_batch, 
                                          total_frames=self._cfg.ddpg.num_iterations*self._cfg.ddpg.data_collector_frames_per_batch,
                                          init_random_frames=self._cfg.ddpg.data_collector_init_frames)

            buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._cfg.ddpg.replay_buffer_capacity),sampler=RandomSampler(), batch_size=self._cfg.ddpg.batch_size)

            self._agents[customer] = {'ddpg': loss_module, 
                                                 'target_updates': target_updates, 
                                                 'optimiser': optimiser,
                                                 'collector': collector,
                                                 'buffer': buffer,
                                                 'envs': envs}
        
    def loadAgent(self, customer):
        return self._agents[customer]['ddpg'], self._agents[customer]['target_updates'], self._agents[customer]['optimiser'], self._agents[customer]['collector'], self._agents[customer]['buffer'], self._agents[customer]['envs']

    def train_eval_test(self):
        for customer in self._customers:
            # Setup progress bar
            pbar = tqdm.tqdm(total=self._cfg.ddpg.num_iterations, desc=f"Customer {customer}")

            # Setup agent
            loss_module, target_updates, optimiser, collector, buffer, envs = self.loadAgent(customer=customer)
            exploration_policy_module = collector.policy

            # Setup evaluation dataframe
            eval_df = []

            # Train
            for iteration, batch in enumerate(collector):
                current_frames = batch.numel()
                buffer.extend(batch)
                # Start training after data_collector_init_frames
                if iteration >= self._cfg.ddpg.data_collector_init_frames:
                    # Train for train_iterations_per_frame iterations per frame
                    for i in range(self._cfg.ddpg.train_iterations_per_frame):
                        sample = buffer.sample()
                        loss = loss_module(sample)
                        loss['loss'].backward()
                        optimiser.step()
                        optimiser.zero_grad()
                        target_updates.step()
                exploration_policy_module[-1].step(current_frames)
                pbar.update(current_frames/self._cfg.ddpg.data_collector_frames_per_batch)

                # Evaluate every eval_period iterations
            #     if iteration % self._cfg.ddpg.eval_period == 0:
            #         eval_env = envs[1]
            #         obs = eval_env.reset()
            #         done = torch.tensor(False)
            #         while(done == torch.tensor(False)):
            #             # print(obs)
            #             action = loss_module.value_network(obs)
            #             # print(action)
            #             obs = eval_env.step(action)['next']
            #             # print(obs)
            #             print("---------------------------")
            #             done = obs['next']['done']
            #             print(done)
            #         eval_df.append({customer: obs['cost'].item()})    
            # # Save evaluation results
            # df = pd.DataFrame(eval_df)
            # self.save_eval(df=df, customer=customer)
            
            # Testing
            test_env = envs[2]
            obs = test_env.reset()
            done = torch.tensor(False)
            data_list = []
            while(done == torch.tensor(False)):
                action = loss_module.actor_network(obs)
                data_list.append({
                    'soe': obs['observation'][0].item(),
                    'net_load': obs['observation'][1].item(),
                    'price': obs['observation'][50].item(),
                    'action': action['action'].item()
                })
                obs = test_env.step(action)['next']
                done = obs['done']

            # Create DataFrame and add calculated columns
            df = pd.DataFrame(data_list)
            df['time'] = df.index
            df['total_load'] = df['net_load'] + df['action']
            df['cost'] = df['price'] * df['total_load']
            df['total_cost'] = df['cost'].cumsum()
            self.save_results(df, customer)
