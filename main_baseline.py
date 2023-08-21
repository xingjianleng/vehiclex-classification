import argparse
import datetime
from distutils.dir_util import copy_tree
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.datasets.vehicle_x import Vehicle_X
from src.models.baseline import LinearNet
from src.utils.logger import Logger


def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('No sys.gettrace')
        is_debug = False

    # transformations
    src_transform = T.Lambda(lambda x: torch.as_tensor(x))
    tgt_transform = T.Lambda(lambda x: x - 1)

    # datasets and dataloaders
    train_set = Vehicle_X(os.path.join(args.base, 'train'), src_transform, tgt_transform)
    val_set = Vehicle_X(os.path.join(args.base, 'val'), src_transform, tgt_transform)
    test_set = Vehicle_X(os.path.join(args.base, 'test'), src_transform, tgt_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model and optimizer
    model = LinearNet(input_dim=2048, output_dim=1362, hidden_dims=[int(x) for x in args.hidden_dims.split(',')]).cuda()

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rprop':
        # TODO: check parameters of Rprop
        optimizer = torch.optim.Rprop(model.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # logging
    logdir = f'logs/{"DEBUG_" if is_debug else ""}BASELINE_' \
             f'lr{args.lr}_b{args.batch_size}_e{args.epochs}_' \
             f'optim{args.optim}_hidden[{args.hidden}]' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}' if not args.eval \
        else f'logs/{args.dataset}/EVAL_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
    os.makedirs(logdir, exist_ok=True)
    copy_tree('src', logdir + '/scripts/src')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print(logdir)
    print('Settings:')
    print(vars(args))

    # training


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Baseline model')
    parser.add_argument('--base', type=str, default='~/Data/vehicle-x/', help='dataset base directory')
    parser.add_argument('--hidden_dims', type=str, default='128,128', help='dimensions of hidden layers, separated by commas')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer',
                        choices=['adam', 'sgd', 'rprop', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--loss', type=str, default='ce', help='loss function',
                        choices=['ce', 'focal'])
    parser.add_argument('--epochs', type=int, default=100, help='number of train epochs')
    parser.add_argument('--log_interval', type=int, default=1000, help='interval of taking the log')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')

    args = parser.parse_args()
    main(args)
