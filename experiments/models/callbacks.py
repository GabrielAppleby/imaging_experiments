import torch
from typing import Optional, Tuple

import torchvision
from pytorch_lightning import Callback, Trainer, LightningModule


class TrainImageReconstructionLogger(Callback):
    def __init__(
        self,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        super().__init__()
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        train_loader = pl_module.train_dataloader()
        batch_size = train_loader.batch_size
        train_sample = next(iter(train_loader))[0]
        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(train_sample.to(pl_module.device))
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(batch_size, *img_dim)

        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

        str_title = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)
