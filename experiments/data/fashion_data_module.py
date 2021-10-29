from typing import Callable

from pl_bolts.datamodules import FashionMNISTDataModule
from torchvision.transforms import Compose, ToTensor, Normalize, Pad


class FashionDataModule(FashionMNISTDataModule):
    dims = (1, 32, 32)

    def default_transforms(self) -> Callable:
        transform_list = [Pad(2), ToTensor()]
        if self.normalize:
            transform_list.append(Normalize(mean=(0.5,), std=(0.5,)))

        return Compose(transform_list)
