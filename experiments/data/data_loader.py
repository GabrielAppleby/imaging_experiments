from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RANDOM_SEED: int = 42
DATA_DIR: Path = Path(Path(__file__).parent.absolute())
RAW_DATA_DIR: Path = Path(DATA_DIR, 'raw')
RECORDS_DATA_DIR: Path = Path(DATA_DIR, 'records')
DUKE_RAW_DATA_DIR: Path = Path(RAW_DATA_DIR, 'duke')
DUKE_DATA_NAME: str = 'duke_oct'
NPZ_NAME_TEMPLATE: str = '{dataset}_{split}.npz'


@dataclass(frozen=True)
class Datasplit:
    """
    A datasplit, consisting of features, labels.
    """
    features: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class Dataset:
    """
    A dataset consisting of train, val, and test.
    """
    train: Datasplit
    val: Datasplit
    test: Datasplit


def load_dataset(dataset_name: str) -> Dataset:
    """
    Loads the processed data set with a projection corresponding to a specific hyperparam and
    fraction of the data.
    :param dataset_name: The data sets' name
    :return: The features, and labels in a Dataset.
    """
    data = np.load(str(Path(DATA_DIR, NPZ_NAME_TEMPLATE.format(dataset_name))), allow_pickle=True)

    features_train = data['features_train'] / np.array(255, dtype=np.float32)
    features_val = data['features_val'] / np.array(255, dtype=np.float32)
    features_test = data['features_test'] / np.array(255, dtype=np.float32)

    train = Datasplit(features_train, data['labels_train'])
    val = Datasplit(features_val, data['labels_val'])
    test = Datasplit(features_test, data['labels_test'])

    return Dataset(train, val, test)


def load_datasplit(dataset_name: str, split: str) -> Datasplit:
    """
    Loads the processed data set with a projection corresponding to a specific hyperparam and
    fraction of the data.
    :param dataset_name: The data sets' name
    :param split: The data splits' name
    :return: The features, and labels in a Dataset.
    """
    data = np.load(
        str(Path(DATA_DIR, NPZ_NAME_TEMPLATE.format(dataset_name, split))), mmap_mode='r')

    split = Datasplit(data['features'], data['labels'])

    return split



def scale_dataset(features: np.ndarray) -> np.ndarray:
    """
    Standardize features by removing the mean and scaling to unit variance.
    :param features: The features matrix of the dataset.
    :return: The scaled features matrix of the dataset.
    """
    return StandardScaler().fit_transform(features)


def normalize_data(data: np.ndarray) -> np.array:
    """
    Normalizes the data to between 0 and 1.
    :param data: The data to be normalized.
    :return: The data, now between 0 and 1.
    """
    min = np.min(data, axis=0)
    data = data - min
    max = np.max(data, axis=0)
    data = np.divide(data, max, out=np.zeros_like(data), where=max != 0)
    return data


def train_val_test_split(features: np.ndarray,
                         labels: np.ndarray) -> Dataset:
    """
    Splits the dataset into training, validation, and test data. 80 percent is used for training,
    10 percent for validation, and 10 percent for testing.
    :param features: The features to split.
    :param labels: The labels to split.
    :return: Training, validation, and test sets of the features, embeddings, and projections.
    """
    features_train, \
    features_both, \
    labels_train, \
    labels_both = train_test_split(
        features, labels, shuffle=True, train_size=0.8, random_state=RANDOM_SEED)

    features_val, \
    features_test, \
    labels_val, \
    labels_test = train_test_split(
        features_both, labels_both, shuffle=True, train_size=0.5, random_state=RANDOM_SEED)
    return Dataset(Datasplit(features_train, labels_train),
                   Datasplit(features_val, labels_val),
                   Datasplit(features_test, labels_test))


def sample_dataset(arr_one: np.ndarray,
                   arr_two: np.ndarray,
                   train_size: float,
                   random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a dataset, taking only a train_size fraction of both arrays given.
    :param arr_one: The first array to sample.
    :param arr_two: The second array to sample.
    :param train_size: The size of the training split.
    :param random_seed: The random state to use.
    :return: The sampled arrays.
    """
    if train_size == 1.0:
        return arr_one, arr_two
    arr_one_sample, _, arr_two_sample, _ = train_test_split(
        arr_one, arr_two, shuffle=True, train_size=train_size, random_state=random_seed)
    return arr_one_sample, arr_two_sample


def create_duke_data(octs_and_label) -> Dataset:
    features = []
    labels = []
    for mat_paths, label in octs_and_label:
        for mat_path in tqdm(mat_paths[:115]):
            mat = loadmat(str(mat_path))['images']
            mat = np.swapaxes(mat, 0, 1)[::2]
            if mat.shape != (500, 512, 100):
                continue
            features.append(mat)
            labels = labels + ([label] * mat.shape[0])
    features = np.concatenate(features, axis=0) / np.array(255, dtype=np.float32)
    labels = np.array(labels)

    return train_val_test_split(features, labels)


def create_duke_data_npz():
    file_ending = '*.mat'

    normal_octs = list(Path(DUKE_RAW_DATA_DIR, 'normal').glob(file_ending))
    amd_octs = list(Path(DUKE_RAW_DATA_DIR, 'amd').glob(file_ending))

    octs_and_label = [(normal_octs, 'normal'), (amd_octs, 'amd')]
    dataset = create_duke_data(octs_and_label)

    np.savez(NPZ_NAME_TEMPLATE.format(dataset=DUKE_DATA_NAME, split='train'),
             features=dataset.train.features,
             labels=dataset.train.labels)
    np.savez(NPZ_NAME_TEMPLATE.format(dataset=DUKE_DATA_NAME, split='val'),
             features=dataset.val.features,
             labels=dataset.val.labels)
    np.savez(NPZ_NAME_TEMPLATE.format(dataset=DUKE_DATA_NAME, split='test'),
             features=dataset.test.features,
             labels=dataset.test.labels)


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    create_duke_data_npz()


    # def array_to_tfrecords(X, y, writer):
    #     feature = {
    #         'feature': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
    #         'label': tf.train.Feature(float_list=tf.train.BytesList(value=y))
    #     }
    #     example = tf.train.Example(features=tf.train.Features(feature=feature))
    #     serialized = example.SerializeToString()
    #     writer.write(serialized)
    #
    # for key, item in asdict(dataset):
    #     filename = Path(RECORDS_DATA_DIR, '{split}.tfrecords'.format(split=key))
    #     writer = tf.python_io.TFRecordWriter(filename)
    #     for feature, label in zip(item.features, item.labels):
    #         array_to_tfrecords(feature, label, writer)
    #     writer.close()

    # np.savez(NPZ_NAME_TEMPLATE.format(dataset=DUKE_DATA_NAME, split='train'),
    #          features=dataset.train.features,
    #          labels=dataset.train.labels)
    # np.savez(NPZ_NAME_TEMPLATE.format(dataset=DUKE_DATA_NAME, split='val'),
    #          features=dataset.val.features,
    #          labels=dataset.val.labels)
    # np.savez(NPZ_NAME_TEMPLATE.format(dataset=DUKE_DATA_NAME, split='test'),
    #          features=dataset.test.features,
    #          labels=dataset.test.labels)
