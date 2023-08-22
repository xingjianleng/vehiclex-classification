import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as T

from src.datasets.vehicle_x import Vehicle_X


if __name__ == '__main__':
    # transformations
    tgt_transform = T.Lambda(lambda x: x - 1)
    train_set = Vehicle_X('/home/lengx/Data/vehicle-x/train', None, tgt_transform)
    val_set = Vehicle_X('/home/lengx/Data/vehicle-x/val', None, tgt_transform)
    test_set = Vehicle_X('/home/lengx/Data/vehicle-x/test', None, tgt_transform)

    plt.figure()
    sns.countplot(x=train_set.labels)
    plt.xticks([])
    plt.title('Train Distribution')
    plt.xlabel('Class')
    plt.savefig('logs/train_dist.png')

    plt.figure()
    sns.countplot(x=val_set.labels)
    plt.xticks([])
    plt.title('Val Distribution')
    plt.xlabel('Class')
    plt.savefig('logs/val_dist.png')

    plt.figure()
    sns.countplot(x=test_set.labels)
    plt.xticks([])
    plt.title('Test Distribution')
    plt.xlabel('Class')
    plt.savefig('logs/test_dist.png')
