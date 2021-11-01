from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor

DATA_DIR: Path = Path('/cluster/tufts/valt/gapple01/imaging/', 'storage')
PNG_DATA_DIR: Path = Path(DATA_DIR, 'png')
CELEBA_PNG_DIR: Path = Path(PNG_DATA_DIR, 'celeba')


class CelebDataModule(pl.LightningDataModule):

    def __init__(self,
                 batch_size: int = 4,
                 resized_shape: Tuple[int, int] = (256, 256),
                 data_dir: str = CELEBA_PNG_DIR,
                 num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.transform = Compose([Resize(resized_shape), ToTensor()])
        self.dims = (3, 256, 256)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: Optional[str] = None):
        dataset = ImageFolder(root=str(CELEBA_PNG_DIR), transform=self.transform)
        num_instances = len(dataset)
        self.train, self.val = random_split(dataset,
                                            [int(num_instances * .9), int(num_instances * .1)])
        self.test = self.val

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
