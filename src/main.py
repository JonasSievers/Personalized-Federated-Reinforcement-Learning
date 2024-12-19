import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
import torch

from utils.LocalLearner import LocalLearner
from utils.FederatedLearner import FederatedLearner

cs = ConfigStore.instance()
cs.store(name="hydra_config", node=HydraConfig)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
    # Device Setup
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA is not supported on this system. Falling back to CPU!')
        device = 'cpu'
    DEVICE = torch.device(device)
    

    if cfg.mode.description == 'local':
        print('Running Local Learner')
        ll = LocalLearner(cfg=cfg, device=DEVICE)
        ll.setup()
        ll.train_eval_test()
    elif cfg.mode.description == 'federated':
        print('Running Federated Learner')
        fl = FederatedLearner(cfg=cfg, device=DEVICE)
        fl.setup()
        fl.train_eval_test()
    elif cfg.mode.description == 'personalized':
        print('Running Personalized Learner')

if __name__ == "__main__":
    main()