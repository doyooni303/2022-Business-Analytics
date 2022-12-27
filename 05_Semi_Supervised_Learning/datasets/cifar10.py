import numpy as np
import torchvision

from .data_utils import Transpose, Normalize


class Labeled_CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root,
        indices=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(Labeled_CIFAR10, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if indices is not None:
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]

        self.data = Transpose(Normalize(self.data))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Unlabeled_CIFAR10(Labeled_CIFAR10):
    def __init__(
        self,
        root,
        indices,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(Unlabeled_CIFAR10, self).__init__(
            root,
            indices,
            train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.targets = np.array([-1 for i in range(len(self.targets))])
