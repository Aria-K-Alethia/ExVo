import pytorch_lightning as pl
import hydra
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from dataset import DataModule
from lightning_module import BaselineLightningModule

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='default')
def train(cfg):
    # loggers
    csvlogger = CSVLogger(save_dir=cfg.train.log_dir, name='csv')
    tblogger = TensorBoardLogger(save_dir=cfg.train.log_dir, name='tb')
    loggers = [csvlogger, tblogger]

    # callbacks
    checkpoint_callback = ModelCheckPoint(dirpath=cfg.train.log_dir, 
                            save_top_k=1, save_last=True,
                            every_n_epochs=1, monitor='val_ccc', mode='max')
    callbacks = [checkpoint_callback]

    datamodule = DataModule(cfg)
    lightning_module = BaselineLightningModule(cfg)
    trainer = pl.Trainer(**cfg.train.trainer, logger=loggers, callbacks=callbacks)
    #trainer.tune(lightning_module, datamodule=datamodule)
    trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == '__main__':
    train()
