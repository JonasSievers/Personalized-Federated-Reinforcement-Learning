import pandas as pd
from utils.Learner import Learner
import utils.utils as utils
import torch
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.modules import Actor, OrnsteinUhlenbeckProcessModule
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer, RandomSampler
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.utils.tensorboard import SummaryWriter
import tqdm



class LocalLearner(Learner):
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)

        
    def train_eval_test(self):
        writer = SummaryWriter(log_dir=f"{self._cfg.output_path}/{self._cfg.name}/")

        for customer in self._customers:
            pbar = tqdm.tqdm(total=self._cfg.ddpg.num_iterations, desc=f"Customer {customer}")

            loss_module, target_updates, optimisers, collector, buffer, envs = self.loadAgent(customer=customer)
            exploration_policy_module = collector.policy

            # Train
            for iteration, batch in enumerate(collector):
                current_frames = batch.numel()
                buffer.extend(batch)
                if iteration >= self._cfg.ddpg.data_collector_init_frames:                     
                    for i in range(self._cfg.ddpg.train_iterations_per_frame):
                        sample = buffer.sample()
                        loss_vals = loss_module(sample)
                        for loss_name in ["loss_actor", "loss_value"]:
                            loss = loss_vals[loss_name]
                            optimiser = optimisers[loss_name]
                            writer.add_scalar(f"{customer}/{loss_name}", loss, iteration*self._cfg.ddpg.train_iterations_per_frame+i)
                            loss.backward()
                            optimiser.step()
                            optimiser.zero_grad()
                        target_updates.step()
                exploration_policy_module[-1].step(current_frames)
                pbar.update(current_frames/self._cfg.ddpg.data_collector_frames_per_batch)

            
            # Testing
            test_env = envs[2]
            obs = test_env.reset()
            done = torch.tensor(False)
            data_list = []
            data_list.append({
                    'soe': obs['observation'][0].item(),
                    'net_load': obs['observation'][1].item(),
                    'price': obs['observation'][26].item(),
                })

            while(done == torch.tensor(False)):
                action = loss_module.actor_network(obs)
                data_list.append({
                    'soe': obs['observation'][0].item(),
                    'net_load': obs['observation'][1].item(),
                    'price': obs['observation'][26].item(),
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
            writer.add_scalar(f"{customer}/cost", test_env.getElectricityCost())
            self._save_results(df, customer)
        writer.close()