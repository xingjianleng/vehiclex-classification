import argparse
import os
import re
import json
import hashlib

import numpy as np
import pandas as pd

from src.utils.argparse_type import str2bool


def extract_info_from_refined_string(s):
    d = {}

    # Extract mode
    nas_retrain = 'NASretrain' in s

    # Extract common hyperparameters
    d['learning_rate'] = float(re.search('lr([\d.]+)', s).group(1))
    d['batch_size'] = int(re.search('_b(\d+)', s).group(1))
    d['epochs'] = int(re.search('_e(\d+)', s).group(1))
    d['weight_decay'] = float(re.search('_wd([\d.]+)', s).group(1))
    d['scheduler'] = str2bool(re.search('_scheduler(\w+)_wd', s).group(1))

    # Extract model specific hyperparameters
    if nas_retrain:
        d['width'] = int(re.search('_width(\d+)_', s).group(1))
        d['cell'] = int(re.search('_cell(\d+)_', s).group(1))
    else:
        d['model'] = re.search('_model(\w+)_scheduler', s).group(1)
        d['pretrained'] = str2bool(re.search('_pretrained(\w+)_', s).group(1))
        d['optimizer'] = re.search('_optim(\w+)_model', s).group(1)

    return hash_hyper(d), d


def hash_hyper(d):
    # Hash hyperparameters to a unique value
    encoded_dict = json.dumps(d, sort_keys=True).encode('utf-8')
    return hashlib.md5(encoded_dict).hexdigest()


def extract_results(fp):
    # Extract results from the test_results.txt file
    return list(map(float, fp.read().split('\n')[:-1]))


def main(args):
    logs = [path for path in sorted(os.listdir(args.log_path)) if os.path.isdir(os.path.join(args.log_path, path))]

    hash_to_hyper = {}
    hash_to_results = {}
    avg_tbl = pd.DataFrame()
    std_tbl = pd.DataFrame()

    # extract hyperparameters and results
    for log in logs:
        hash, d = extract_info_from_refined_string(log)
        if hash not in hash_to_hyper:
            hash_to_hyper[hash] = d
            result = {
                'acc': [],
                'prec': [],
                'rec': [],
                'f1': [],
            }
            hash_to_results[hash] = result
        with open(os.path.join(args.log_path, log, 'test_results.txt'), 'r') as fp:
            result = extract_results(fp)
            hash_to_results[hash]['acc'].append(result[0])
            hash_to_results[hash]['prec'].append(result[1])
            hash_to_results[hash]['rec'].append(result[2])
            hash_to_results[hash]['f1'].append(result[3])

    # create table with average results and standard deviation
    for hash, results in hash_to_results.items():
        hyper = hash_to_hyper[hash]
        tbl = pd.DataFrame(results)
        avg_row = tbl.mean()
        std_row = tbl.std()
        avg_tbl = pd.concat([avg_tbl, pd.DataFrame([avg_row])], ignore_index=True)
        std_tbl = pd.concat([std_tbl, pd.DataFrame([std_row])], ignore_index=True)

    # create output directory
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # sort based on accuracy (descending)
    idx_sorted = np.argsort(avg_tbl['acc'].values)[::-1]
    avg_tbl = avg_tbl.iloc[idx_sorted].reset_index(drop=True)
    std_tbl = std_tbl.iloc[idx_sorted].reset_index(drop=True)

    # save tables
    avg_tbl.to_csv(os.path.join(args.out_path, f'average_metrics.csv'))
    std_tbl.to_csv(os.path.join(args.out_path, f'std_metrics.csv'))

    # save hyperparameters
    hypers = []
    for i in range(len(avg_tbl)):
        hyper = hash_to_hyper[list(hash_to_hyper.keys())[idx_sorted[i]]]
        hypers.append(hyper)

    # save hyperparameters as a json file
    with open(os.path.join(args.out_path, f'configs.json'), 'w') as fp:
        json.dump(hypers, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Results parser')
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--out_path', type=str, default='./out/')

    args = parser.parse_args()
    main(args)
