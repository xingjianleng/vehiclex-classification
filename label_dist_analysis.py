from collections import Counter
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from src.datasets.vehicle_x import Vehicle_X


if __name__ == '__main__':
    # transformations
    train_set = Vehicle_X(os.path.expanduser('~/Data/vehicle-x_v2/Classification Task/train'))
    val_set = Vehicle_X(os.path.expanduser('~/Data/vehicle-x_v2/Classification Task/val'))
    test_set = Vehicle_X(os.path.expanduser('~/Data/vehicle-x_v2/Classification Task/test'))

    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')

    plt.figure()
    sns.countplot(x=train_set.labels, palette='hls')
    plt.xticks([])
    plt.xlabel('Class')
    plt.tight_layout()
    plt.savefig('logs/train_dist.pdf')

    plt.figure()
    sns.countplot(x=val_set.labels, palette='hls')
    plt.xticks([])
    plt.xlabel('Class')
    plt.tight_layout()
    plt.savefig('logs/val_dist.pdf')

    plt.figure()
    sns.countplot(x=test_set.labels, palette='hls')
    plt.xticks([])
    plt.xlabel('Class')
    plt.tight_layout()
    plt.savefig('logs/test_dist.pdf')

    train_counter = Counter(train_set.labels)
    val_counter = Counter(val_set.labels)
    test_counter = Counter(test_set.labels)

    with open('logs/label_dist.txt', 'w') as fp:
        fp.write('Train set:\n')
        label_counts = np.array(list(train_counter.values()))
        fp.write(f'Max_count: {label_counts.max()}; Min_count: {label_counts.min()}\n')
        fp.write(f'Max_label: {label_counts.argmax()}; Min_label: {label_counts.argmin()}\n')
        fp.write(f'Total: {label_counts.sum()}; Mean: {label_counts.mean()}; Std: {label_counts.std()}\n\n')
        
        fp.write('Val set:\n')
        label_counts = np.array(list(val_counter.values()))
        fp.write(f'Max_count: {label_counts.max()}; Min_count: {label_counts.min()}\n')
        fp.write(f'Max_label: {label_counts.argmax()}; Min_label: {label_counts.argmin()}\n')
        fp.write(f'Total: {label_counts.sum()}; Mean: {label_counts.mean()}; Std: {label_counts.std()}\n\n')

        fp.write('Test set:\n')
        label_counts = np.array(list(test_counter.values()))
        fp.write(f'Max_count: {label_counts.max()}; Min_count: {label_counts.min()}\n')
        fp.write(f'Max_label: {label_counts.argmax()}; Min_label: {label_counts.argmin()}\n')
        fp.write(f'Total: {label_counts.sum()}; Mean: {label_counts.mean()}; Std: {label_counts.std()}\n\n')
