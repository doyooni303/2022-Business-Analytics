import numpy as np
import torch


class RandomFlip(object):
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class RandomPadandCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        old_h, old_w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, old_h - new_h)
        left = np.random.randint(0, old_w - new_w)

        x = x[:, top : top + new_h, left : left + new_w]
        return x


class Transform_Twice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        out1 = self.transform(img)
        out2 = self.transform(img)
        return out1, out2


def Normalize(x, m=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2345, 0.2616)):
    x, m, std = [np.array(a, np.float32) for a in (x, m, std)]
    x -= m * 255
    x *= 1.0 / (255 * std)
    return x


def Transpose(x, source="NHWC", target="NCHW"):
    return x.transpose([source.index(d) for d in target])


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode="reflect")
