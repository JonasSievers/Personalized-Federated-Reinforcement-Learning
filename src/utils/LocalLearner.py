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
import tqdm
import numpy as np
import pandas as pd
import os



class LocalLearner():
    def __init__(self, cfg: dict, device: torch.device):
        self._cfg = cfg #Stores configuration  with env and ddpg details
        self._customers = cfg.env.customer #Array of customers
        self._device = device #cpu
        self._agents = {} #Dic for all modules (actor, critic, optimizer, buffer, etc)
        self._datasets = utils.createDatasets(self._cfg) #Loads train, eval, test datasets and replaces price with new price data
        self.patience = cfg.ddpg.early_stopping_patience  # Number of evaluations to wait for improvement


    def _save_model_state(self, customer, loss_module):
        """Save the model state for a given customer."""
        output_path = f"{self._cfg.output_path}/{self._cfg.name}"
        os.makedirs(output_path, exist_ok=True)
        
        state_dict = {
            'actor': loss_module.actor_network.state_dict(),
            'critic': loss_module.value_network.state_dict(),
        }
        torch.save(state_dict, f"{output_path}/customer_{customer}_best_model.pth")   
    
    def _load_model_state(self, customer, loss_module):
        """Load the best model state for a given customer."""
        output_path = f"{self._cfg.output_path}/{self._cfg.name}"
        model_path = f"{output_path}/customer_{customer}_best_model.pth"
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, weights_only=True)
            loss_module.actor_network.load_state_dict(state_dict['actor'])
            loss_module.value_network.load_state_dict(state_dict['critic'])

    def setup(self):
        self._observation_spec, self._action_spec = utils.createEnvSpecs(self._cfg) #Define shape and type of observation/actions in env

        for customer in self._customers: #For each building
            envs = utils.create_envs(customer, self._datasets, self._cfg, self._device) #Setup the train, eval, test environments

            actor_net, critic_net = utils.createActorCritic(self._cfg) #Create Actor and Critic Nets (MLP)
            policy_module = Actor(module=actor_net, in_keys=["observation"], out_keys=["action"]) #Torch Wrapper
            critic_module = TensorDictModule(module=critic_net, in_keys=["observation", "action"], out_keys=["state_action_value"]) #Torch Wrapper

            loss_module = DDPGLoss(actor_network=policy_module, value_network=critic_module, delay_actor=True, delay_value=True) #Loss 
            loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self._cfg.ddpg.td_gamma) #Value estimator

            target_updates = SoftUpdate(loss_module=loss_module, tau=self._cfg.ddpg.target_update_tau) #Soft target updates

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

    def _save_results(self, df, customer):
        output_path = f"{self._cfg.output_path}/{self._cfg.name}"
        os.makedirs(output_path, exist_ok=True)
        filename = f"{output_path}/building_{customer}_results.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def _collect_test_data(self, test_env, loss_module):
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
        return df

    def train_eval_test_local(self):
        #Sets up TensorBoard logging directory for storing training/evaluation metrics.
        writer = SummaryWriter(log_dir=f"{self._cfg.output_path}/{self._cfg.name}/")

        for customer in self._customers:
            pbar = tqdm.tqdm(total=self._cfg.ddpg.num_iterations, desc=f"Customer {customer}") #Progressbar visualization

            #Retrieve modules
            loss_module = self._agents[customer]['ddpg']
            target_updates = self._agents[customer]['target_updates']
            optimisers = self._agents[customer]['optimisers']
            collector = self._agents[customer]['collector']
            exploration_policy_module = self._agents[customer]['collector'].policy
            buffer = self._agents[customer]['buffer']

            # Early stopping variables
            best_eval_cost = -float('inf')
            best_iteration = 0
            patience_counter = 0

            for iteration, batch in enumerate(collector): #Each iteration returns a batch of transitions collected by SyncDataCollector
                current_frames = batch.numel() #Retrieves how many frames are in this batch.
                buffer.extend(batch) #Stores the batch in the replay buffer.

                if iteration >= self._cfg.ddpg.data_collector_init_frames: #Check if Enough Frames have been Collected
                    
                    if iteration % self._cfg.ddpg.eval_period == 0: #Evaluate every X iterations
                        eval_iteration = iteration // self._cfg.ddpg.eval_period
                        eval_env = self._agents[customer]['envs'][1] #picks the evaluation environment.
                        obs = eval_env.reset() #starts from initial state.
                        done = torch.tensor(False)
                        while not done: #Step through the environment with the actor’s deterministic policy (no OU noise).
                            action = loss_module.actor_network(obs)
                            obs = eval_env.step(action)['next']
                            done = obs['done']
                        #Log the resulting electricity cost
                        current_eval_cost = float(eval_env.getElectricityCost())
                        writer.add_scalar(f"{customer}/eval", eval_env.getElectricityCost(), eval_iteration)

                        # Early stopping check
                        if current_eval_cost > best_eval_cost: #Positive -> revenue
                            #print(f"Updated best model (Current: {current_eval_cost} / Best: {best_eval_cost})")
                            best_eval_cost = current_eval_cost
                            best_iteration = iteration
                            patience_counter = 0
                            self._save_model_state(customer, loss_module)  # Save the best model state
                        else:
                            #print(f"No improvement {patience_counter} (Current: {current_eval_cost} / Best: {best_eval_cost})")
                            patience_counter += 1

                        # If no improvement for patience iterations, stop training
                        if patience_counter >= self.patience:
                            #print(f"\nEarly stopping triggered for customer {customer} at iteration {iteration}")
                            #print(f"Best evaluation cost: €{float(best_eval_cost):.2f} at iteration {best_iteration}")
                            break
                    
                    for i in range(self._cfg.ddpg.train_iterations_per_frame): #Training iterations per batch
                        sample = buffer.sample() #Sample a batch from the replay buffer
                        loss_vals = loss_module(sample)
                        
                        for loss_name in ["loss_actor", "loss_value"]:
                            loss = loss_vals[loss_name]
                            optimiser = optimisers[loss_name]
                            writer.add_scalar(f"{customer}/{loss_name}", loss, iteration*self._cfg.ddpg.train_iterations_per_frame+i)
                            loss.backward() #Backpropagate
                            optimiser.step() #Update parameters
                            optimiser.zero_grad() #Reset gradients
                        target_updates.step() #Perform the soft update of the target networks.
                exploration_policy_module[-1].step(current_frames) #Update exploration noise
                pbar.update(current_frames/self._cfg.ddpg.data_collector_frames_per_batch)
            
            # Load the best model for testing
            self._load_model_state(customer, loss_module)
            
            #Testing
            test_env = self._agents[customer]['envs'][2]
            test_data = self._collect_test_data(test_env, loss_module)
            self._save_results(test_data, customer)
            writer.add_scalar(f"{customer}/cost", test_env.getElectricityCost())
        writer.close()