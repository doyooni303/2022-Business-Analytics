import numpy as np
import torchvision

from .cifar10 import Labeled_CIFAR10, Unlabeled_CIFAR10
from .data_utils import Transform_Twice


def split_datasets(labels, n_labeled_per_class, n_valid=100):
    labels = np.array(labels, dtype=int)
    indice_labeled, indice_unlabeled, indice_val = (
        [],
        [],
        [],
    )  # labeled, unlabeled, validation data 분류

    for i in range(10):  # Number of labels in CIFAR10
        indice_tmp = np.where(labels == i)[0]

        # validation: 클래스 별 100개 labeled
        # train: labeled-n_labeled_per_class 만큼 / unlabeled-나머지

        indice_labeled.extend(indice_tmp[:n_labeled_per_class])
        indice_unlabeled.extend(indice_tmp[n_labeled_per_class:n_valid])
        indice_val.extend(indice_tmp[-n_valid:])

    for i in [indice_labeled, indice_unlabeled, indice_val]:
        np.random.shuffle(i)

    return indice_labeled, indice_unlabeled, indice_val


def get_cifar10(
    data_dir: str,
    n_labeled: int,
    n_valid: int,
    transform_train=None,
    transform_val=None,
    download=True,
):

    base_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=download)

    indice_labeled, indice_unlabeled, indice_val = split_datasets(
        base_dataset.targets, int(n_labeled / 10), n_valid
    )

    train_labeled_set = Labeled_CIFAR10(
        data_dir, indice_labeled, train=True, transform=transform_train
    )

    train_unlabeled_set = Unlabeled_CIFAR10(
        data_dir,
        indice_unlabeled,
        train=True,
        transform=Transform_Twice(transform_train),
    )

    val_set = Labeled_CIFAR10(
        data_dir, indice_val, train=True, transform=transform_val, download=True
    )  # validation dataset

    test_set = Labeled_CIFAR10(
        data_dir, train=False, transform=transform_val, download=True
    )  # test dataset

    return train_labeled_set, train_unlabeled_set, val_set, test_set
