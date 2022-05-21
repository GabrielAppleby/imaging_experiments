import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from data.celeb_data_module import CelebDataModule
from models.callbacks import TrainImageReconstructionLogger
from models.hvae import VAE


def main():
    dm: pl.LightningDataModule = CelebDataModule(num_workers=4, batch_size=1)
    dm.prepare_data()
    dm.setup()
    train(dm)


def train(dm: LightningDataModule):
    callbacks = [TrainImageReconstructionLogger(),
                 LearningRateMonitor(logging_interval='epoch')]

    model = VAE(input_shape=dm.dims, num_hidden_channels=12)
    trainer = Trainer(gpus=0,
                      max_time="00:47:50:00",
                      max_epochs=300,
                      callbacks=callbacks)
    trainer.fit(model, datamodule=dm)
    torch.save(model, 'vae.pt')


if __name__ == '__main__':
    main()
