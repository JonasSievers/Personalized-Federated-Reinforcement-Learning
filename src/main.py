import torch
import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig

import logging
logger = logging.getLogger(__name__)
from learner.LocalLearner import LocalLearner



@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):
     # Device Setup
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.info('Warning: CUDA is not supported on this system. Falling back to CPU!')
        device = 'cpu'
    DEVICE = torch.device(device)
    logger.info(f'{device} initialized!')

    # Set Seed
    torch.manual_seed(cfg.seed)
    logger.info(f'Seed: {cfg.seed} initialized!')

    logger.info(f'Start pipeline {cfg.component.name} for building id {cfg.building_id}')
    
    match cfg.component.name:
        case 'local':
            ll = LocalLearner(cfg=cfg, device=DEVICE)
            ll.setup()
            ll.train()

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="hydra_config", node=HydraConfig) 
    main()