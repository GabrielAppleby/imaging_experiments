from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchvision.datasets import CelebA
from torchvision.transforms import Compose, Resize, ToTensor

DATA_DIR: Path = Path(Path(Path(__file__).parent.absolute()), 'storage')
PNG_DATA_DIR: Path = Path(DATA_DIR, 'torch')
CELEBA_PNG_DIR: Path = Path(PNG_DATA_DIR, 'celeba')


class CelebDataModule(pl.LightningDataModule):

    def __init__(self,
                 batch_size: int = 4,
                 resized_shape: Tuple[int, int] = (224, 224),
                 data_dir: str = CELEBA_PNG_DIR):
        super().__init__()
        self.batch_size = batch_size
        self.transform = Compose([Resize(resized_shape), ToTensor()])
        self.dims = resized_shape
        self.data_dir = data_dir
        self.num_workers = 4
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        CelebA(root=str(CELEBA_PNG_DIR), split='all', transform=self.transform, download=True)

    def setup(self, stage: Optional[str] = None):
        self.train = CelebA(root=str(CELEBA_PNG_DIR), split='train', transform=self.transform)
        self.val = CelebA(root=str(CELEBA_PNG_DIR), split='val', transform=self.transform)
        self.test = CelebA(root=str(CELEBA_PNG_DIR), split='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)
