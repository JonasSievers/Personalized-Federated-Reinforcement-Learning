import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
import torch
import os
import shutil


from utils.LocalLearner import LocalLearner
from utils.FederatedLearner import FederatedLearner
from utils.PersonalizedLearner import PersonalizedLearner
from utils.Forecaster import Forecaster

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
    os.makedirs(f"{cfg.model_path}/{cfg.name}", exist_ok=True)
    os.makedirs(f"{cfg.output_path}/forecaster", exist_ok=True)
    os.makedirs(f"{cfg.model_path}/forecaster", exist_ok=True)
    
    # Forecaster Setup
    if cfg.use_forecast:
        not_exist = False
        for customer in cfg.env.customer:
            not_exist = not_exist or not (os.path.exists(f"{cfg.model_path}/forecaster/{customer}.pt"))

        if not not_exist:
            print("All Forecaster already trained!")
        else:
            print("Training Forecaster!")
            shutil.rmtree(f"{cfg.model_path}/forecaster")
            os.makedirs(f"{cfg.model_path}/forecaster", exist_ok=True)
            for customer in cfg.env.customer:
                forecaster = Forecaster(cfg=cfg, customer=customer, mode='train', calc_metric=True)
                forecaster.train(epochs=100)
                forecaster.evaluate()
                
    
    match cfg.mode.description:
        case 'local':
            print('Running Local Learner')
            ll = LocalLearner(cfg=cfg, device=DEVICE)
            ll.train()
            # ll.test()
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