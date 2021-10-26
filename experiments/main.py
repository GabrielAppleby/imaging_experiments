from pathlib import Path

import torch
from typing import Tuple

import torchvision
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, GPUStatsMonitor

from data.duke_data_module import DukeDataModule
from pl_bolts.datamodules import FashionMNISTDataModule
from models.callbacks import TrainImageReconstructionLogger
from models.hvae import VAE

RESULTS_DIR: Path = Path(Path(Path(__file__).parent.absolute()), 'results')


def transform():
    return torchvision.transforms.Compose([torchvision.transforms.Pad(2), torchvision.transforms.ToTensor()])


def main():
    image_shape = (1, 32, 32)

    # dm = DukeDataModule(resized_shape=(256, 256), batch_size=1)
    dm = FashionMNISTDataModule(num_workers=4)
    dm.default_transforms = transform
    dm.prepare_data()
    dm.setup()

    train(image_shape, dm)


def train(image_shape: Tuple[int, int, int], dm: LightningDataModule):
    callbacks = [EarlyStopping(monitor='val_loss', patience=10), TrainImageReconstructionLogger(), GPUStatsMonitor()]

    model = VAE(input_shape=image_shape)
    trainer = Trainer(gpus=1,
                      max_time="00:23:50:00",
                      max_epochs=50,
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
