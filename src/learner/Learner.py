import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MLP, OrnsteinUhlenbeckProcessModule, TanhModule
from torchrl.objectives import ValueEstimators, SoftUpdate, DDPGLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, RandomSampler

from utils import make_env, make_dataset


class Learner():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.DEVICE = device
            
    def setup(self):
        train_dataset = make_dataset(cfg=self.cfg, mode=self.cfg.mode, device=self.DEVICE)
        spec_env =  make_env(cfg=self.cfg, dataset=train_dataset, device=self.DEVICE)
        action_spec = spec_env.action_spec
        observation_spec = spec_env.observation_spec['observation']
    
        policy_net = MLP(
            in_features=observation_spec.shape[-1],
            out_features=action_spec.shape[-1],
            num_cells=self.cfg.ddpg.policy.num_cells,
            activation_class=torch.nn.ReLU,
            device=self.DEVICE
        )
          
        policy_module = TensorDictModule(
            module=policy_net,
            in_keys=['observation'],
            out_keys=['action']
        )

        actor = TensorDictSequential(
            policy_module,
            TanhModule(
                spec=action_spec,
                in_keys=['action'],
                out_keys=['action'],
            ),
        )

        ou_module = OrnsteinUhlenbeckProcessModule(
            annealing_num_steps=self.cfg.ddpg.ou.annealing_num_steps,
            n_steps_annealing=self.cfg.ddpg.ou.n_steps_annealing,
            spec=action_spec,
            device=self.DEVICE
        )
        
        exploration_policy = TensorDictSequential(
            actor,
            ou_module
        )

        critic = TensorDictModule(
            module=MLP(
                in_features=observation_spec.shape[-1] + action_spec.shape[-1],
                out_features=1,
                depth=2,
                num_cells=self.cfg.ddpg.critic.num_cells,
                activation_class=torch.nn.ReLU,
                device=self.DEVICE
            ),
            in_keys=['observation', 'action'],
            out_keys=['state_action_value']
        )
        
        self.collector = SyncDataCollector(
            create_env_fn=(make_env(cfg=self.cfg, dataset=train_dataset, device=self.DEVICE)),
            policy=exploration_policy,
            frames_per_batch=self.cfg.ddpg.collector_frames_per_batch,
            total_frames=-1,
            device=self.DEVICE)
        
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.ddpg.replay_buffer_size, device=self.DEVICE),
            sampler=RandomSampler(),
            batch_size=self.cfg.ddpg.batch_size)

        self.loss_module = DDPGLoss(
            actor_network=actor,
            value_network=critic,
            delay_actor=self.cfg.ddpg.delay_actor,
            delay_value=self.cfg.ddpg.delay_value).to(device=self.DEVICE)
        
        self.loss_module.make_value_estimator(
            value_type=ValueEstimators.TD0,
            gamma=self.cfg.ddpg.gamma)

        self.target_updater = SoftUpdate(
            loss_module=self.loss_module, 
            tau=self.cfg.ddpg.tau)

        self.optimiser_dict = {
            'loss_actor': torch.optim.Adam(params=self.loss_module.actor_network.parameters(), lr=self.cfg.ddpg.lr.actor),
            'loss_value': torch.optim.Adam(params=self.loss_module.value_network.parameters(), lr=self.cfg.ddpg.lr.value)
        }

    def train(self):
        pass

    def val(self, loss_module):
        with torch.no_grad():
            val_dataset = make_dataset(cfg=self.cfg, mode='val', device=self.DEVICE)
            val_env =  make_env(cfg=self.cfg, dataset=val_dataset, device=self.DEVICE)
            val_env.base_env.eval()
            tensordict_result = val_env.rollout(max_steps=100000, policy=loss_module.actor_network)
            return tensordict_result
        
    def test(self):
        pass