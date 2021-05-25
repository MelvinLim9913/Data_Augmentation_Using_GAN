import os
import glob

import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),  # latent dimension = 100
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod((3, 255, 255)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *(3, 255, 255))
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod((3, 255, 255))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.image_paths = glob.glob(os.path.join(path, '*.png'))
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)
