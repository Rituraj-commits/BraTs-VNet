import numpy as np
import torch


class RandomFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    def random_flip(self, img, label):
        axes = [0, 1, 2]
        rand = np.random.randint(0, 3)
        img = self.flip_axis(img, axes[rand])
        img = np.squeeze(img)

        if label is None:
            return img
        else:
            label = self.flip_axis(label, axes[rand])
            label = np.squeeze(label)
        return img, label

    def forward(self, img, label):
        if(torch.rand(1) < self.p):
            img, label = self.random_flip(img, label)
            return img, label
        return img, label
