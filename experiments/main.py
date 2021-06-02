from pathlib import Path

import torch
from typing import Tuple

import torchvision
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping

from data.duke_data_module import DukeDataModule
from models.ae import AE, VAE
from models.callbacks import TrainImageReconstructionLogger

RESULTS_DIR: Path = Path(Path(Path(__file__).parent.absolute()), 'results')


def main():
    image_shape = (224, 224)

    dm = DukeDataModule(resized_shape=image_shape, batch_size=64)
    dm.prepare_data()
    dm.setup()

    train(image_shape, dm)


def train(image_shape: Tuple[int, int], dm: LightningDataModule):
    callbacks = [EarlyStopping(monitor='val_loss', patience=10), TrainImageReconstructionLogger()]

    model = VAE(input_height=image_shape[0], enc_type='resnet50', enc_out_dim=2048, latent_dim=512)
    trainer = Trainer(gpus=1,
                      max_epochs=500,
                      callbacks=callbacks)
    trainer.fit(model, datamodule=dm)
    torch.save(model, 'vae.pt')


def compare_decoded_images(dm: LightningDataModule):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = torch.load('vae.pt')
    trainer = dm.train_dataloader()
    for idx, batch in enumerate(trainer):
        real_images = batch[0]
        fake_images = model(real_images.to(model.device))
        torchvision.utils.save_image(real_images, Path(RESULTS_DIR, '{}_real.png'.format(idx)))
        torchvision.utils.save_image(fake_images, Path(RESULTS_DIR, '{}_fake.png'.format(idx)))


if __name__ == '__main__':
    main()
