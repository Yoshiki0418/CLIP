import wandb
from omegaconf import OmegaConf, DictConfig

class WandBMetricsWriter():
    def __init__(
        self,
        project_name: str,
        model_name: str = None,
    ) -> None:
        self.project_name = project_name
        self.name = model_name


        wandb.init(project=project_name,entity="yoshi-ai", name=self.name)

    def __call__(
            self, 
            epoch: int, 
            train_loss: float, 
            train_top_k_acc: float,
            val_loss: float,
            val_top_k_acc: float,
        ) -> None:
        wandb.log(
            {"train_loss": train_loss,
             "train_top_k_acc": train_top_k_acc,
             "val_loss": val_loss,
             "val_top_k_acc": val_top_k_acc,
             }, step=epoch)

    def finish(self) -> None:
        wandb.finish()