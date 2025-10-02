import torch
from tensordict import TensorDict 

import logging
logger = logging.getLogger(__name__)
import os

from learner.Learner import Learner

class LocalLearner(Learner):
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        
    def train(self):
        exploration_policy = self.collector.policy

        best_iteration = TensorDict({
                'iteration': 0,
                'value': 10000000,
                'td': None
        })

        for iteration, batch in enumerate(self.collector):
            current_frames = batch.numel()
            exploration_policy[-1].step(current_frames)
            self.replay_buffer.extend(batch)

            sample = self.replay_buffer.sample()
            loss_vals = self.loss_module(sample)

            for loss_name in ["loss_actor", "loss_value"]:
                optimiser = self.optimiser_dict[loss_name]
                optimiser.zero_grad()
                loss = loss_vals[loss_name]
                loss.backward()
                optimiser.step()
                self.target_updater.step()

            if (iteration+1) % self.cfg.ddpg.eval_interval == 0:
                tensordict_result = self.val(self.loss_module)
                final_cost = torch.sum(tensordict_result['next']['cost'], dim=0)
                if final_cost < best_iteration['value']:
                    best_iteration['iteration'] = iteration
                    best_iteration['value'] = final_cost
                    logger.info(f'Iteration: {iteration}, lowest cost: {final_cost.item()}')
                    os.makedirs(f'{self.cfg.model_path}/', exist_ok=True)
                    torch.save(self.loss_module.state_dict(), f'{self.cfg.model_path}/loss_module.pth')

            if iteration - best_iteration['iteration'] > 2000:
                metrics = {
                    'best_iteration': best_iteration['iteration'],
                    'final_cost': best_iteration['value'],
                }
                torch.save(metrics, f'{self.cfg.output_path}/metrics.pt')
                
                return best_iteration['value']