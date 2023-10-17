'''
Code is useful when users download the dataset from the official website.
'''
import argparse
import json
import os


def main(args):
    dataset_path = os.path.expanduser(args.dataset_path)

    train_names = [name.split('.')[0] for name in sorted(os.listdir(os.path.join(dataset_path, 'train')))]
    val_names = [name.split('.')[0] for name in sorted(os.listdir(os.path.join(dataset_path, 'val')))]
    test_names = [name.split('.')[0] for name in sorted(os.listdir(os.path.join(dataset_path, 'test')))]

    with open(os.path.join(dataset_path, 'split_cfg.json'), 'w') as f:
        json.dump({'train': train_names, 'val': val_names, 'test': test_names}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate split for vehicle-x dataset')
    parser.add_argument('--dataset_path', type=str, default='~/Data/vehicle-x/', help='Path to dataset')
    args = parser.parse_args()

    main(args)
