import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, GPUStatsMonitor

from data.celeb_data_module import CelebDataModule
from models.callbacks import TrainImageReconstructionLogger
from models.hvae import VAE


def main():
    dm: pl.LightningDataModule = CelebDataModule(num_workers=4)
    dm.prepare_data()
    dm.setup()
    train(dm)


def train(dm: LightningDataModule):
    callbacks = [EarlyStopping(monitor='val_loss', patience=10), TrainImageReconstructionLogger(), GPUStatsMonitor()]

    model = VAE(input_shape=dm.dims)
    trainer = Trainer(gpus=1,
                      max_time="00:23:50:00",
                      max_epochs=50,
                      callbacks=callbacks)
    trainer.fit(model, datamodule=dm)
    torch.save(model, 'vae.pt')


if __name__ == '__main__':
    main()
