import argparse
import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.datasets.vehicle_x import Vehicle_X


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculate mean and std of dataset')
    parser.add_argument('--base', type=str, default='~/Data/vehicle-x_v2/Classification Task/train/',
                        help='dataset base directory')
    args = parser.parse_args()

    # calculate the mean and std of the vehicle-x training dataset
    dataset = Vehicle_X(os.path.expanduser(args.base), transform=T.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # bs=1 to avoid problem when calculating std

    # Initialize the variables
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std.. <==')

    for inputs, _ in dataloader:
        for i in range(3): # 3 channels: R, G, B
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()

    # Final mean and std
    mean.div_(len(dataloader))
    std.div_(len(dataloader))

    print(f'mean: {mean}; std: {std}')
