import pandas as pd
from utils.Learner import Learner
import torch
import tqdm



class LocalLearner(Learner):
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
    def train_eval_test(self):
        for customer in self._customers:
            # Setup progress bar
            pbar = tqdm.tqdm(total=self._cfg.ddpg.num_iterations, desc=f"Customer {customer}")

            # Setup agent
            loss_module, target_updates, optimisers, collector, buffer, envs = self.loadAgent(customer=customer)
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
                        loss_vals = loss_module(sample)
                        for loss_name in ["loss_actor", "loss_value"]:
                            loss = loss_vals[loss_name]
                            optimiser = optimisers[loss_name]
                            loss.backward()
                            optimiser.step()
                            optimiser.zero_grad()
                        target_updates.step()
                exploration_policy_module[-1].step(current_frames)
                pbar.update(current_frames/self._cfg.ddpg.data_collector_frames_per_batch)

                # Evaluate every eval_period iterations
                if iteration % self._cfg.ddpg.eval_period == 0:
                    eval_env = envs[1]
                    obs = eval_env.reset()
                    done = torch.tensor(False)
                    while(done == torch.tensor(False)):
                        action = loss_module.actor_network(obs)
                        obs = eval_env.step(action)['next']
                        done = obs['done']
                    eval_df.append({customer: obs['cost'].item()})    
            # Save evaluation results
            df = pd.DataFrame(eval_df)
            self.save_eval(df=df, customer=customer)
            
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
