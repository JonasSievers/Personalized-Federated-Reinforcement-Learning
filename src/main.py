import torch
import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
from envs.EnergyManagementEnv import EnergyManagementEnv
from utils.EnergyDataset import EnergyDataset
from utils.Networks import CustomActorNet, CustomCriticNet
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs, RandomPolicy
from torchrl.envs.transforms import TransformedEnv, InitTracker
from torchrl.modules import Actor, OrnsteinUhlenbeckProcessModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer, RandomSampler
from torchrl.data import Bounded, Unbounded, Composite

from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

from torch.utils.tensorboard import SummaryWriter


import tqdm


cs = ConfigStore.instance()
cs.store(name="hydra_config", node=HydraConfig)

def create_env(train_dataset, cfg, days=None):
    return TransformedEnv(EnergyManagementEnv(customer=1, dataset=train_dataset, cfg=cfg.env_params, test_days=days), InitTracker())

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    train_dataset = EnergyDataset("data/Final_Energy_dataset.csv",'train')
    test_dataset = EnergyDataset("data/Final_Energy_dataset.csv",'test')


    writer = SummaryWriter(log_dir=f"{cfg.experiment.path}/{cfg.experiment.name}")
   
    # 
    env = create_env(train_dataset,cfg)
    test_env = create_env(train_dataset=test_dataset,cfg=cfg, days=cfg.env_params.test_days)

    actor_net = CustomActorNet(env.observation_spec, env.action_spec, fc_layers=(400,300,))
    policy_module = Actor(module=actor_net, in_keys=["observation"], out_keys=["action"])
    ou = OrnsteinUhlenbeckProcessModule(spec=policy_module.spec.clone(), theta=0.15, sigma=0.2, annealing_num_steps=1000)
    exploration_policy_module = TensorDictSequential(policy_module, ou)

    critic_net = CustomCriticNet(env.observation_spec, env.action_spec, obs_fc_layers=(400,), joint_fc_layers=(300,))
    critic_module = TensorDictModule(module=critic_net, in_keys=["observation", "action"], out_keys=["state_action_value"])

    collector = SyncDataCollector(create_env_fn=create_env(train_dataset,cfg),policy=exploration_policy_module, frames_per_batch=100, total_frames=200_000)
    buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=1000),sampler=RandomSampler())


    loss_module = DDPGLoss(
        actor_network=policy_module,  # Use the non-explorative policies
        value_network=critic_module,
        # delay_actor=True,  # Whether to use a target network for the actor
        delay_value=True,  # Whether to use a target network for the value
        loss_function="l2",
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=0.99)

    target_updates = SoftUpdate(loss_module=loss_module, tau=0.001)

    actor_optimizer = torch.optim.Adam(loss_module.actor_network.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(loss_module.value_network.parameters(), lr=1e-3)

    optimisers = {"loss_actor": actor_optimizer, "loss_value": critic_optimizer}


    for iteration, batch in enumerate(collector):
        buffer.extend(batch)

        sample = buffer.sample(64)

        loss_vals = loss_module(sample)
        # print(loss_vals['loss_actor'])

        for loss_name in ["loss_actor", "loss_value"]:
            loss = loss_vals[loss_name]
            optimiser = optimisers[loss_name]
            writer.add_scalar(f"Loss/{loss_name}", loss, iteration)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        if iteration % 5 == 0:
            target_updates.step()
        if iteration % 100 == 0:
            obs = test_env.reset()

            done = torch.tensor(False)
            while(done == torch.tensor(False)):
                action = policy_module(obs)
                obs = test_env.step(action)['next']
                done = obs['done']
            print(f"Iteration {iteration}: { test_env.printPrice()}")


    obs = test_env.reset()

    done = torch.tensor(False)
    while(done == torch.tensor(False)):
        action = policy_module(obs)
        obs = test_env.step(action)['next']
        done = obs['done']
    print(test_env.printPrice())
    
    writer.close()
    return

if __name__ == "__main__":
    main()