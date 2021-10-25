import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Resize

from data.duke_data_module import DUKE_PNG_DIR


def main():
    dataset = ImageFolder(root=str(DUKE_PNG_DIR),
                          transform=Compose([Grayscale(), Resize((32, 32))]))
    arrays = []
    labels = []
    for img, label in dataset:
        arrays.append(np.array(img).reshape((1, -1)))
        labels.append(label)
    mat = np.concatenate(arrays)
    for k in [2] + list(range(5, 30, 5)):
        mapper = umap.UMAP(n_neighbors=k).fit(mat)
        umap.plot.points(mapper, labels=np.array(labels))
        plt.savefig('{}.png'.format(k))


if __name__ == '__main__':
    main()
