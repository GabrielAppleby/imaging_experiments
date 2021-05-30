from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from PIL import Image

RANDOM_SEED: int = 42
DATA_DIR: Path = Path(Path(Path(__file__).parent.absolute()), 'storage')
RAW_DATA_DIR: Path = Path(DATA_DIR, 'storage/raw')
PNG_DATA_DIR: Path = Path(DATA_DIR, 'storage/png')
DUKE_RAW_DIR: Path = Path(RAW_DATA_DIR, 'duke')
DUKE_RAW_NORMAL_DIR: Path = Path(DUKE_RAW_DIR, 'normal')
DUKE_RAW_AMD_DIR: Path = Path(DUKE_RAW_DIR, 'amd')

DUKE_PNG_DIR: Path = Path(PNG_DATA_DIR, 'duke')
DUKE_PNG_NORMAL_DIR: Path = Path(DUKE_PNG_DIR, 'normal')
DUKE_PNG_AMD_DIR: Path = Path(DUKE_PNG_DIR, 'amd')


def main():
    DUKE_PNG_NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    DUKE_PNG_AMD_DIR.mkdir(parents=True, exist_ok=True)
    file_ending = '*.mat'

    raw_normal_octs = list(DUKE_RAW_NORMAL_DIR.glob(file_ending))
    raw_amd_octs = list(DUKE_RAW_AMD_DIR.glob(file_ending))

    oct_paths = [(raw_normal_octs, DUKE_PNG_NORMAL_DIR), (raw_amd_octs, DUKE_PNG_AMD_DIR)]
    for mat_paths, png_path in oct_paths:
        idx = 0
        for mat_path in tqdm(mat_paths[:115]):
            mat = loadmat(str(mat_path))['images']
            mat = np.swapaxes(mat, 0, 2)
            mat = np.swapaxes(mat, 2, 1)[::2]
            if mat.shape != (50, 512, 1000):
                continue
            for image in mat:
                im = Image.fromarray(image)
                im.save(Path(png_path, "oct_{}.jpg".format(idx)))
                idx += 1


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    main()
