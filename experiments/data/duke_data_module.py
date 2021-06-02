from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale

DATA_DIR: Path = Path(Path(Path(__file__).parent.absolute()), 'storage')
PNG_DATA_DIR: Path = Path(DATA_DIR, 'png')
DUKE_PNG_DIR: Path = Path(PNG_DATA_DIR, 'duke')


class DukeDataModule(pl.LightningDataModule):

    def __init__(self,
                 batch_size: int = 4,
                 resized_shape: Tuple[int, int] = (224, 224),
                 data_dir: str = DUKE_PNG_DIR):
        super().__init__()
        self.batch_size = batch_size
        # Histogram normalization,
        self.transform = Compose([Grayscale(), Resize(resized_shape), ToTensor()])
        self.dims = resized_shape
        self.data_dir = data_dir
        self.num_workers = 1
        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: Optional[str] = None):
        dataset = ImageFolder(root=str(DUKE_PNG_DIR), transform=self.transform)
        num_instances = len(dataset)
        self.train, self.val, self.test = random_split(dataset,
                                                       [int(num_instances * .6),
                                                        int(num_instances * .2),
                                                        int(num_instances * .2)])

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
