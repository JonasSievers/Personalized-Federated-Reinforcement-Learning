import pandas as pd
from utils.Learner import Learner
import torch
import tqdm


class LocalLearner(Learner):
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
    def train(self):
        eval_df = pd.DataFrame(columns=self._customers)
        for customer in self._customers:
            eval = {}
            # Setup progress bar
            pbar = tqdm.tqdm(total=self._cfg.algorithm.num_iterations, desc=f"Customer {customer}")

            # Setup agent
            loss_module, target_updates, optimiser, collector, buffer, envs = self._loadAgent(customer=customer)
            exploration_policy_module = collector.policy

            # Train
            for iteration, batch in enumerate(collector):
                current_frames = batch.numel()
                exploration_policy_module[-1].step(current_frames)
                buffer.extend(batch)

                # Train for train_iterations_per_frame iterations per frame
                for i in range(self._cfg.algorithm.train_iterations_per_frame):
                    sample = buffer.sample()
                    match self._algorithm:
                        case 'ddpg':
                            loss_vals = loss_module(sample)
                            for loss_name in ["loss_actor", "loss_value"]:
                                loss = loss_vals[loss_name]
                                loss.backward()
                                optimiser = optimiser[loss_name]
                                optimiser.step()
                                optimiser.zero_grad()
                        case 'dqn':
                            loss = loss_module(sample)
                            loss['loss'].backward()
                            optimiser.step()
                            optimiser.zero_grad()
                    if (iteration+i) % self._cfg.algorithm.target_update_period == 0:
                            target_updates.step()

                # Update train progress bar
                pbar.update(current_frames/self._cfg.algorithm.data_collector_frames_per_batch)
            
                # Evaluate every eval_period iterations
                if (iteration+1) % self._cfg.algorithm.eval_period == 0:
                    eval_env = envs[2]
                    eval_env.reset()
                    match self._algorithm:
                        case 'ddpg':
                            tensordict_result = eval_env.rollout(max_steps=eval_env.get_max_timesteps(), policy=loss_module.actor_network)
                        case 'dqn':
                            tensordict_result = eval_env.rollout(max_steps=eval_env.get_max_timesteps(), policy=loss_module.value_network)
                    final_cost = tensordict_result[-1]['next']['cost']
                    eval[iteration+1] = final_cost.item()

            # Save evaluation metrics
            eval_df[customer] = eval

            # Save the model
            match self._algorithm:
                case 'ddpg':
                    model_path = f"{self._cfg.model_path}/actor_network_{customer}.pt"
                    torch.save(loss_module.actor_network.state_dict(), model_path)
                case 'dqn':
                    model_path = f"{self._cfg.model_path}/value_network_{customer}.pt"
                    torch.save(loss_module.value_network.state_dict(), model_path)
        
        # Save evaluation DataFrame to CSV
        eval_df.to_csv(f"{self._cfg.output_path}/{self._cfg.name}/eval_metrics.csv", index=False)