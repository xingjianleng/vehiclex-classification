import argparse
import datetime
from distutils.dir_util import copy_tree
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import tqdm

from src.trainer import NetworkTrainer
from src.datasets.vehicle_x import Vehicle_X
from src.models.timm_wrapper import get_model
from src.utils.draw_curve import draw_curve
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

    # get models and transformations from timm, and transfer to GPU
    if not args.nas:
        model, train_transform, test_transform = get_model(args.model, args.num_classes, pretrained=args.pretrained)
        model.cuda()
    else:
        raise NotImplementedError('NAS not implemented')

    # datasets and dataloaders
    train_set = Vehicle_X(os.path.expanduser(os.path.join(args.base, 'train')), transform=train_transform)
    val_set = Vehicle_X(os.path.expanduser(os.path.join(args.base, 'val')), transform=test_transform)
    test_set = Vehicle_X(os.path.expanduser(os.path.join(args.base, 'test')), transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.tr_batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.te_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.te_batch_size, shuffle=False, num_workers=args.num_workers)

    # optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {args.optim} not implemented')

    # logging
    logdir = f'{args.logdir}{"DEBUG_" if is_debug else ""}{"NAS" if args.nas else "BASELINE"}_' \
             f'lr{args.lr}_b{args.tr_batch_size}_e{args.epochs}_' \
             f'optim{args.optim}_model{args.model}_scheduler{args.scheduler}' \
             f'_wd{args.weight_decay}_es{args.early_stop}_seed{args.seed}_pretrained{args.pretrained}_' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}' if not args.eval \
        else f'{args.logdir}{args.dataset}/EVAL_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
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

    # tensorboard logging
    writer = SummaryWriter(logdir)
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                    for key, value in vars(args).items()])))

    # training
    trainer = NetworkTrainer(model, logdir, writer, args)

    # scheduler
    def warmup_lr_scheduler(epoch, warmup_epochs=0.1 * args.epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler)
    else:
        scheduler = None

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_acc_s = []
    val_loss_s = []
    val_acc_s = []

    # early stop and best acc record
    best_acc = 0
    early_stop = 0

    # learning
    if not args.eval:
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            train_loss, train_acc = trainer.train(epoch, train_loader, optimizer, scheduler)

            # log train results
            val_loss, val_acc, _, _, _ = trainer.test(val_loader, epoch)
            # draw & save
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_acc_s.append(train_acc)
            val_loss_s.append(val_loss)
            val_acc_s.append(val_acc)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, val_loss_s,
                        train_acc_s, val_acc_s)

            # early stop
            if val_acc > best_acc:
                best_acc = val_acc
                early_stop = 0
                # only save the best model
                torch.save(model.state_dict(), os.path.join(logdir, 'model_best.pth'))
            else:
                early_stop += 1
            if early_stop >= args.early_stop:
                print('Early stop!')
                break

        # load best model for testing
        print(f'Finished training, loading best model for testing, best acc: {best_acc:.6f}')
        model.load_state_dict(torch.load(os.path.join(logdir, 'model_best.pth')))

    print('Test loaded model...')
    print(logdir)
    test_results = trainer.test(test_loader)
    with open(os.path.join(logdir, 'test_results.txt'), 'w') as f:
        for test_result in test_results[1:]:
            f.write(f'{test_result:.6f}\n')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main experiment script')
    # dataset
    parser.add_argument('--base', type=str, default='~/Data/vehicle-x_v2/Classification Task/', help='dataset base directory')
    parser.add_argument('--num_classes', type=int, default=1362, help='number of classes')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--tr_batch_size', '-b', type=int, default=64, help='train batch size')
    parser.add_argument('--te_batch_size', type=int, default=128, help='test batch size')
    # model parameter
    parser.add_argument('--model', type=str, default='resnet50', help='model name')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--nas', action='store_true', help='use neural architecture search (NAS)')
    # training parameter
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of train epochs')
    # optimizer parameter
    parser.add_argument('--optim', type=str, default='adam', help='optimizer',
                        choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--scheduler', action='store_true', help='enable lambda scheduler')
    # other parameters
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--logdir', type=str, default='./logs/', help='log directory')
    parser.add_argument('--log_interval', type=int, default=200, help='interval of epochs of taking the log')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    main(args)
