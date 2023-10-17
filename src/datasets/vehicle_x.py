import os

from PIL import Image
from torchvision.datasets import VisionDataset


class Vehicle_X(VisionDataset):
    # dataset class for vehicle-x, which is responsible for loading images and labels
    def __init__(self, root, transform=None, target_transform=None):
        super(Vehicle_X, self).__init__(root, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.img_names = []
        self.labels = []

        for img_name in os.listdir(root):
            self.img_names.append(img_name)
            # labels are stored as 0001-1362, index minus 1 to make it 0-1361
            self.labels.append(int(img_name.split('_')[0]) - 1)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        label = self.labels[index]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)
