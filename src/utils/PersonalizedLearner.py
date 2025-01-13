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
import os
import yaml
import omegaconf
import pandas as pd
import tqdm


class PersonalizedLearner():
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
        # Iterate through each customer
        for customer in self._customers:
            # Extract all necessary components for training
            loss_module = self._agents[customer]['ddpg']  # DDPG agent
            target_updates = self._agents[customer]['target_updates']  # For updating target networks
            optimisers = self._agents[customer]['optimisers']  # Optimizers for actor and critic
            collector = self._agents[customer]['collector']  # Data collector for experiences
            exploration_policy_module = self._agents[customer]['collector'].policy  # Exploration policy
            buffer = self._agents[customer]['buffer']  # Replay buffer

            # Collect and process batches of transitions
            for iteration, batch in enumerate(collector):  
                current_frames = batch.numel()  # Get number of frames in current batch
                buffer.extend(batch)  # Add batch to replay buffer

                # Only start training after collecting enough initial frames
                if global_iteration >= self._cfg.ddpg.data_collector_init_frames:
                    # Periodic evaluation
                    if global_iteration % self._cfg.ddpg.eval_period == 0:
                        eval_iteration = iteration // self._cfg.ddpg.eval_period
                        eval_env = self._agents[customer]['envs'][1]  # Get evaluation environment
                        obs = eval_env.reset()  # Reset environment
                        done = torch.tensor(False)
                        
                        # Run evaluation episode
                        while not done:
                            action = loss_module.actor_network(obs)  # Get action from actor
                            obs = eval_env.step(action)['next']  # Take step in environment
                            done = obs['done']
                        
                        # Log evaluation results
                        writer.add_scalar(f"{customer}/eval", eval_env.getElectricityCost(), eval_iteration)

                    # Training loop
                    for i in range(self._cfg.ddpg.train_iterations_per_frame):
                        sample = buffer.sample()  # Sample batch from replay buffer
                        loss_vals = loss_module(sample)  # Compute losses
                        
                        # Update both actor and critic networks
                        for loss_name in ["loss_actor", "loss_value"]:
                            loss = loss_vals[loss_name]
                            optimiser = optimisers[loss_name]
                            
                            # Log losses
                            writer.add_scalar(
                                f"{customer}/{loss_name}", 
                                loss, 
                                (global_iteration+iteration)*self._cfg.ddpg.train_iterations_per_frame+i
                            )
                            
                            # Gradient update
                            loss.backward()
                            optimiser.step()
                            optimiser.zero_grad()
                        
                        target_updates.step()  # Update target networks
                    
                # Update exploration policy
                exploration_policy_module[-1].step(current_frames)

    def _aggregate(self):
        # Initialize empty lists to store parameters from all customers
        self._actor_params_list = []
        self._critic_params_list = []
        
        # Iterate through each customer in the system
        for customer in self._customers:
            # Get the actor network parameters from current customer's DDPG agent
            # state_dict() returns a dictionary containing model's weights and biases
            actor_params = self._agents[customer]['ddpg'].actor_network_params.state_dict()
            self._actor_params_list.append(actor_params)
            
            # Get the critic/value network parameters from current customer's DDPG agent
            # These parameters represent how the network evaluates actions
            critic_params = self._agents[customer]['ddpg'].value_network_params.state_dict()
            self._critic_params_list.append(critic_params)
    
    def _federatedAveraging(self):
        # Initialize empty dictionaries for global model parameters
        self._global_actor_params = {}
        self._global_critic_params = {}

        # Get all parameter keys
        actor_keys = list(self._actor_params_list[0].keys())[:-2]  # Exclude batch_size and device
        critic_keys = list(self._critic_params_list[0].keys())[:-2]

        # Process Actor Network Parameters
        # Only average the first n layers specified in config
        for key in actor_keys[:self._cfg.personalized_fed.num_shared_layers]:
            self._global_actor_params[key] = torch.stack([params[key] for params in self._actor_params_list]).mean(dim=0)
        
        # Copy the remaining layers from first client (they stay fixed)
        for key in actor_keys[self._cfg.personalized_fed.num_shared_layers:]:
            self._global_actor_params[key] = self._actor_params_list[0][key]

        # Copy metadata
        self._global_actor_params['__batch_size'] = self._actor_params_list[0]['__batch_size']
        self._global_actor_params['__device'] = self._actor_params_list[0]['__device']

        # Process Critic Network Parameters
        # Similar process for critic network
        for key in critic_keys[:self._cfg.personalized_fed.num_shared_layers]:
            self._global_critic_params[key] = torch.stack([params[key] for params in self._critic_params_list]).mean(dim=0)
        
        # Copy fixed layers from first client
        for key in critic_keys[self._cfg.personalized_fed.num_shared_layers:]:
            self._global_critic_params[key] = self._critic_params_list[0][key]

        # Copy metadata
        self._global_critic_params['__batch_size'] = self._critic_params_list[0]['__batch_size']
        self._global_critic_params['__device'] = self._critic_params_list[0]['__device']

    def _updateLocal(self):
        # Get parameter keys
        actor_keys = list(self._global_actor_params.keys())[:-2]  # Exclude batch_size and device
        critic_keys = list(self._global_critic_params.keys())[:-2]

        for customer in self._customers:
            # Get current parameters
            current_actor_params = self._agents[customer]['ddpg'].actor_network_params.state_dict()
            current_critic_params = self._agents[customer]['ddpg'].value_network_params.state_dict()

            # Create new parameter dictionaries
            new_actor_params = {}
            new_critic_params = {}

            # Update shared layers from global model
            for key in actor_keys[:self._cfg.personalized_fed.num_shared_layers]:
                new_actor_params[key] = self._global_actor_params[key]
            
            # Keep local parameters for non-shared layers
            for key in actor_keys[self._cfg.personalized_fed.num_shared_layers:]:
                new_actor_params[key] = current_actor_params[key]

            # Same for critic network
            for key in critic_keys[:self._cfg.personalized_fed.num_shared_layers]:
                new_critic_params[key] = self._global_critic_params[key]
            
            for key in critic_keys[self._cfg.personalized_fed.num_shared_layers:]:
                new_critic_params[key] = current_critic_params[key]

            # Copy metadata
            new_actor_params['__batch_size'] = self._global_actor_params['__batch_size']
            new_actor_params['__device'] = self._global_actor_params['__device']
            new_critic_params['__batch_size'] = self._global_critic_params['__batch_size']
            new_critic_params['__device'] = self._global_critic_params['__device']

            # Load the new parameters
            self._agents[customer]['ddpg'].actor_network_params.load_state_dict(new_actor_params)
            self._agents[customer]['ddpg'].value_network_params.load_state_dict(new_critic_params)

        
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
    
    def _save_results(self, df, customer):
        output_path = f"{self._cfg.output_path}/{self._cfg.name}"
        os.makedirs(output_path, exist_ok=True)
        filename = f"{output_path}/building_{customer}_results.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def train_eval_test(self):
        # Initialize logging and progress tracking
        writer = SummaryWriter(log_dir=f"{self._cfg.output_path}/{self._cfg.name}/")  # TensorBoard writer
        pbar = tqdm.tqdm(
            total=self._cfg.federated.num_fed_rounds, 
            desc='Federeated Learning'
        )  # Progress bar for tracking
        
        # Main Federated Learning Loop
        for i in range(self._cfg.federated.num_fed_rounds):
            # Update progress bar description with current round
            pbar.set_description(f"Federated Learning Round {i+1}")
            
            # Calculate global iteration number
            global_iteration = (i+1) * self._cfg.federated.num_local_rounds
            
            # Execute one round of federated learning:
            self._trainLocal(global_iteration, writer)      # Train each client locally
            self._aggregate()                               # Collect parameters from all clients
            self._federatedAveraging()                      # Average the parameters
            self._updateLocal()                             # Update all clients with new parameters
            
            pbar.update(1)  # Update progress bar
        
        # Save final global parameters
        save_path = f"{self._cfg.output_path}/{self._cfg.name}"
        torch.save(self._global_actor_params, f"{save_path}/global_actor_params.pth")
        torch.save(self._global_critic_params, f"{save_path}/global_critic_params.pth")
        # Save config file
        # Convert Hydra config to dictionary
        config_dict = omegaconf.OmegaConf.to_container(self._cfg, resolve=True)

        # Save as yaml
        with open(f"{save_path}/config.yaml", 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print("Saved Global Parameters")

        # Testing Phase - For each customer
        for customer in self._customers:
            # Get required components for testing
            loss_module = self._agents[customer]['ddpg']
            test_env = self._agents[customer]['envs'][2]  # Get test environment
            
            # Initialize test environment
            obs = test_env.reset()
            done = torch.tensor(False)
            
            # Initialize storage for trajectory
            obs_history = []
            action_history = []
            
            # Run single test episode
            while(done == torch.tensor(False)):
                obs_history.append(obs)                    # Store observation
                action = loss_module.actor_network(obs)    # Get action from policy
                action_history.append(action)              # Store action
                obs = test_env.step(action)['next']       # Take step in environment
                done = obs['done']                        # Check if episode is done
            
            # Perform final testing and save results
            test_env = self._agents[customer]['envs'][2]  # Reset test environment
            test_data = self._collect_test_data(test_env, loss_module)  # Collect test data
            self._save_results(test_data, customer)  # Save test results
            
            # Log final electricity cost
            writer.add_scalar(f"{customer}/cost", test_env.getElectricityCost())
        
        # Close the TensorBoard writer
        writer.close()
