import hydra
from hydra.core.config_store import ConfigStore
from conf.config import HydraConfig
from utils.LocalLearner import LocalLearner



import tqdm

from utils.Networks import CustomActorNet, CustomCriticNet


cs = ConfigStore.instance()
cs.store(name="hydra_config", node=HydraConfig)



@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: HydraConfig):

    ll = LocalLearner(customers=cfg.experiment.env_params.customer, cfg=cfg)
    ll.setup()
    ll.train_eval_test()


    return

if __name__ == "__main__":
    main()