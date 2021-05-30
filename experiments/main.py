import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from data.duke_data_module import DukeDataModule
from models.ae import AE
from models.callbacks import TrainImageReconstructionLogger


def main():
    image_shape = (32, 32)
    callbacks = [EarlyStopping(monitor='val_loss', patience=15), TrainImageReconstructionLogger()]

    dm = DukeDataModule(resized_shape=image_shape, batch_size=64)
    dm.prepare_data()
    dm.setup()
    model = AE(input_height=image_shape[0])
    trainer = Trainer(gpus=1,
                      max_epochs=100,
                      callbacks=callbacks)
    trainer.fit(model, datamodule=dm)
    torch.save(model, 'vae.pt')


if __name__ == '__main__':
    main()
