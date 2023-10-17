'''
Code is useful when users download the dataset from the official website.
'''
import argparse
import json
import os
import shutil


def main(args):
    dataset_path = os.path.expanduser(args.dataset_path)
    img_path = os.path.join(dataset_path, 'VeRi_ReID_Simulation')
    with open(os.path.expanduser(args.cfg_path), 'r') as f:
        cfg = json.load(f)
    
    for key, value in cfg.items():
        if not os.path.exists(os.path.join(dataset_path, key)):
            os.makedirs(os.path.join(dataset_path, key))
        else:
            print('Folder {} already exists, skipping...'.format(key))
            continue
        for name in value:
            shutil.copy(os.path.join(img_path, name + '.jpg'), os.path.join(dataset_path, key, name + '.jpg'))

    print('Verifying split...')
    print('Train: {}'.format(len(os.listdir(os.path.join(dataset_path, 'train')))))
    print('Val: {}'.format(len(os.listdir(os.path.join(dataset_path, 'val')))))
    print('Test: {}'.format(len(os.listdir(os.path.join(dataset_path, 'test')))))
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Utility to split dataset into train, val, test')
    parser.add_argument('--dataset_path', type=str, default='~/Data/VeRi_ReID_Simulation/', help='Path to dataset')
    parser.add_argument('--cfg_path', type=str, default='~/Data/vehicle-x/split_cfg.json', help='Path to split config')
    args = parser.parse_args()

    main(args)
