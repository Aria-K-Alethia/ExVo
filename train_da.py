import pytorch_lightning as pl
import hydra
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from dataset import DataModule, DADataModule
from lightning_module import BaselineLightningModule, DALightningModule

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='default')
def train(cfg):
    # loggers
    csvlogger = CSVLogger(save_dir=cfg.train.log_dir, name='csv')
    tblogger = TensorBoardLogger(save_dir=cfg.train.log_dir, name='tb')
    loggers = [csvlogger, tblogger]

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.train.log_dir, 
                            save_top_k=1, save_last=True,
                            every_n_epochs=1, monitor='val_ccc', mode='max')
    earlystop_callback = EarlyStopping(monitor='val_ccc', min_delta=1e-4,
                            patience=5, mode='max', check_finite=True,
                            stopping_threshold=1.0, divergence_threshold=0.0)
    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint_callback, earlystop_callback, lr_monitor]

    datamodule = DADataModule(cfg)
    lightning_module = DALightningModule(cfg)
    trainer = pl.Trainer(**cfg.train.trainer, default_root_dir=hydra.utils.get_original_cwd(),
                    logger=loggers, callbacks=callbacks)
    #trainer.tune(lightning_module, datamodule=datamodule)
    trainer.fit(lightning_module, datamodule=datamodule)
    print(f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    train()
