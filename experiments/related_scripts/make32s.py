from pathlib import Path

import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Resize

from data.duke_data_module import DUKE_PNG_DIR


SAVE_PATH = '/home/gabriel/Documents/embedding_combiner/frontend/duke/'

def main():
    dataset = ImageFolder(root=str(DUKE_PNG_DIR),
                          transform=Compose([Resize((32, 32))]))
    for data, img_name in zip(dataset, [x[0].split('/')[-1] for x in dataset.samples]):
        img = data[0]
        label = data[1]
        if label == 0:
            img_type = 'amd'
        else:
            img_type = 'normal'
        path = '{}_{}'.format(img_type, img_name)
        img.save(Path(SAVE_PATH, path))



if __name__ == '__main__':
    main()
