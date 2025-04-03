import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
import torch
import os

from utils.LocalLearner import LocalLearner
from utils.FederatedLearner import FederatedLearner
from utils.PersonalizedLearner import PersonalizedLearner

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

    # Folder Setup
    os.makedirs(cfg.output_path, exist_ok=True)
    os.makedirs(cfg.model_path, exist_ok=True)
    
    match cfg.mode.description:
        case 'local':
            print('Running Local Learner')
            ll = LocalLearner(cfg=cfg, device=DEVICE)
            ll.train()
        # case 'federated':
        #     print('Running Federated Learner')
        #     fl = FederatedLearner(cfg=cfg, device=DEVICE)
        #     fl.setup()
        #     fl.train_eval_test()
        # case 'personalized':
        #     print('Running Personalized Learner')
        #     pfl = PersonalizedLearner(cfg=cfg, device=DEVICE)
        #     pfl.setup()
        #     pfl.train_eval_test()

if __name__ == "__main__":
    main()