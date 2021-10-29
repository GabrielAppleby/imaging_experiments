import torch
import torchvision

from pathlib import Path

from pytorch_lightning import LightningDataModule

RESULTS_DIR: Path = Path(Path(Path(__file__).parent.absolute()), 'results')


def compare_decoded_images(dm: LightningDataModule):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = torch.load('vae.pt')
    trainer = dm.train_dataloader()
    for idx, batch in enumerate(trainer):
        real_images = batch[0]
        fake_images = model(real_images.to(model.device))
        torchvision.utils.save_image(real_images, Path(RESULTS_DIR, '{}_real.png'.format(idx)))
        torchvision.utils.save_image(fake_images, Path(RESULTS_DIR, '{}_fake.png'.format(idx)))