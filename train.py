import pytorch_lightning as pl
import hydra
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from dataset import DataModule
from lightning_module import BaselineLightningModule

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='default')
def train(cfg):
    datamodule = DataModule(cfg)
    lightning_module = BaselineLightningModule(cfg)
    trainer = pl.Trainer(gpus=1,  max_epochs=10, min_epochs=5)
    trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == '__main__':
    train()
