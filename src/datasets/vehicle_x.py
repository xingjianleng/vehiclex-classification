import os

import numpy as np
from torchvision.datasets import VisionDataset


class Vehicle_X(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(Vehicle_X, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        # store attributes
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # I/O is slow, put data loading in initialization
        self.feats = []
        self.labels = []
        filenames = sorted(os.listdir(root))
        for filename in filenames:
            if filename.endswith('.npy'):
                feat = np.load(os.path.join(root, filename))
                self.feats.append(feat)
                self.labels.append(int(filename.split('_')[0]))
    
    def __getitem__(self, index):
        feat = self.feats[index]
        label = self.labels[index]
        if self.transform is not None:
            feat = self.transform(feat)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return feat, label

    def __len__(self):
        return len(self.feats)
