import argparse
import datetime
from distutils.dir_util import copy_tree
import os
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import tqdm

from src.trainer import NetworkTrainer
from src.datasets.vehicle_x import Vehicle_X
from src.models.baseline import LinearNet
from src.utils.logger import Logger
from src.utils.draw_curve import draw_curve


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
    train_set = Vehicle_X(os.path.expanduser(os.path.join(args.base, 'train')), src_transform, tgt_transform)
    val_set = Vehicle_X(os.path.expanduser(os.path.join(args.base, 'val')), src_transform, tgt_transform)
    test_set = Vehicle_X(os.path.expanduser(os.path.join(args.base, 'test')), src_transform, tgt_transform)

    if args.batch_size < 1:
        args.train_batch_size = len(train_set)
        args.val_batch_size = len(val_set)
        args.test_batch_size = len(test_set)
    else:
        args.train_batch_size = args.batch_size
        args.val_batch_size = args.batch_size
        args.test_batch_size = args.batch_size

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    # model and optimizer
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')] if args.hidden_dims else []
    model = LinearNet(input_dim=2048, output_dim=1362, hidden_dims=hidden_dims).cuda()

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rprop':
        # for Rprop, the lr is set to 0.01 (initial step size), eta and weight step are default values
        optimizer = torch.optim.Rprop(model.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # logging
    logdir = f'logs/{"DEBUG_" if is_debug else ""}BASELINE_' \
             f'lr{args.lr}_b{args.batch_size}_e{args.epochs}_' \
             f'optim{args.optim}_hidden[{args.hidden_dims}]_scheduler{args.scheduler}' \
             f'_loss{args.loss}_gamma{args.gamma}_wd{args.weight_decay}_' \
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
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'cosine_restart':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 10)
        elif args.scheduler == 'lambda':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler)
        else:
            raise NotImplementedError(f'Scheduler {args.schduler} not implemented')
    else:
        scheduler = None

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_acc_s = []
    test_loss_s = []
    test_acc_s = []

    # learning
    if not args.eval:
        # trainer.test(test_loader)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            train_loss, train_acc = trainer.train(epoch, train_loader, optimizer, scheduler)
            if epoch % args.log_epoch == 0:
                test_loss, test_acc = trainer.test(val_loader, epoch)
                print(f'Epoch {epoch + 1}/{args.epochs}\tTrain loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, '
                      f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')

                # draw & save
                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                train_acc_s.append(train_acc)
                test_loss_s.append(test_loss)
                test_acc_s.append(test_acc)
                draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s,
                           train_acc_s, test_acc_s)
                torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))

    print('Test loaded model...')
    print(logdir)
    trainer.test(test_loader)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Baseline model')
    # dataset
    parser.add_argument('--base', type=str, default='~/Data/vehicle-x/', help='dataset base directory')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size')
    # training parameter
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of train epochs')
    parser.add_argument('--loss', type=str, default='ce', help='loss function',
                        choices=['ce', 'focal'])
    parser.add_argument('--gamma', type=float, default=2, help='gamma for focal loss')
    # network parameter
    parser.add_argument('--hidden_dims', type=str, default='', help='dimensions of hidden layers, separated by commas')
    # optimizer parameter
    parser.add_argument('--optim', type=str, default='adam', help='optimizer',
                        choices=['adam', 'sgd', 'rprop', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--scheduler', type=str, default=None, help='scheduler',
                        choices=['cosine', 'cosine_restart', 'lambda'])
    # other parameters
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--log_epoch', type=int, default=25, help='interval of epochs of taking the log')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    main(args)
