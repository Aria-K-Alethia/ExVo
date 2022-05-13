import pytorch_lightning as pl
import hydra
from torch.utils.data import DataLoader
from dataset import ExvoDataset
from lightning_module import BaselineLightningModule

@hydra.main(config_path='config', config_name='default')
def train(cfg):
    dataset = ExvoDataset(**cfg.train_dataset)
    train_loader = DataLoader(dataset, 8, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    lightning_module = BaselineLightningModule(cfg)
    trainer = pl.Trainer(gpus=1,  max_epochs=10, min_epochs=5)
    trainer.fit(lightning_module, train_dataloaders=train_loader)


if __name__ == '__main__':
    train()
