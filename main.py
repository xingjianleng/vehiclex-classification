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
from src.models.baseline import LinearNet
from src.models.constr_casc import ConstructiveCascadeNetwork
from src.utils.draw_curve import draw_curve
from src.utils.logger import Logger
from src.utils.constr_casc_util import update_optimizer


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
    actviations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
    if args.constr_casc:
        assert len(hidden_dims) == 1, 'Allow exactly one initial hidden layer for constructive cascade network'
        learning_rates = {'L1': args.lr,
                          'L2': args.lr * args.l2_ratio,
                          'L3': args.lr * args.l3_ratio}
        model = ConstructiveCascadeNetwork(input_dim=2048, output_dim=1362, initial_hidden_dim=hidden_dims[0],
                                        activation_fn=actviations[args.activation], weight_init=args.weight_init).cuda()
    else:
        model = LinearNet(input_dim=2048, output_dim=1362, hidden_dims=hidden_dims,
                        activation_fn=actviations[args.activation], weight_init=args.weight_init).cuda()

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rprop':
        # for Rprop, eta and weight step are default values
        optimizer = torch.optim.Rprop(model.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {args.optim} not implemented')

    # logging
    cascade_info = f'_casc_hidden{args.cascade_hidden_dim}_casc_' \
                   f'dropout{args.cascade_dropout}_casc_thresh{args.threshold}' \
                   f'_max_casc_layer{args.max_cascade_layers}' \
                   f'_l2{args.l2_ratio}_l3{args.l3_ratio}_'
    logdir = f'{args.logdir}{"DEBUG_" if is_debug else ""}{"CONSTR_CASC" if args.constr_casc else "BASELINE"}_' \
             f'lr{args.lr}_b{args.batch_size}_e{args.epochs}_' \
             f'optim{args.optim}_hidden[{args.hidden_dims}]_scheduler{args.scheduler}' \
             f'_loss{args.loss}_gamma{args.gamma}_wd{args.weight_decay}' \
             f'{cascade_info if args.constr_casc else "_"}'\
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
    val_loss_s = []
    val_acc_s = []

    # learning
    if not args.eval:
        # trainer.test(test_loader)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            train_loss, train_acc = trainer.train(epoch, train_loader, optimizer, scheduler)

            # Constructive Cascade Network add cascade layer
            if args.constr_casc and train_loss < args.threshold and \
                    (args.max_cascade_layers is None or len(model.cascade_layers) < args.max_cascade_layers):
                model.add_cascade_layer(args.cascade_hidden_dim, args.cascade_dropout)
                update_optimizer(optimizer, model, learning_rates)
                print('New cascade layer added, total hidden dimension: ', model.total_hidden_dim)

            if epoch % args.log_epoch == 0:
                val_loss, val_acc = trainer.test(val_loader, epoch)
                # draw & save
                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                train_acc_s.append(train_acc)
                val_loss_s.append(val_loss)
                val_acc_s.append(val_acc)
                draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, val_loss_s,
                           train_acc_s, val_acc_s)
                torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))

    print('Test loaded model...')
    print(logdir)
    _, test_acc = trainer.test(test_loader)
    with open(os.path.join(logdir, 'test_acc.txt'), 'w') as f:
        f.write(f'{test_acc:.6f}')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main experiment script')
    # dataset
    parser.add_argument('--base', type=str, default='~/Data/vehicle-x/', help='dataset base directory')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--batch_size', '-b', type=int, default=0, help='batch size')
    # training parameter
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of train epochs')
    parser.add_argument('--loss', type=str, default='ce', help='loss function',
                        choices=['ce', 'focal'])
    parser.add_argument('--gamma', type=float, default=2, help='gamma for focal loss')
    # network parameter
    parser.add_argument('--hidden_dims', type=str, default='128', help='dimensions of hidden layers, separated by commas')
    parser.add_argument('--activation', type=str, default='tanh', help='activation function',
                        choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--weight_init', type=str, default='xavier', help='weight initialization',
                        choices=['xavier', 'kaiming'])
    # optimizer parameter
    parser.add_argument('--optim', type=str, default='adam', help='optimizer',
                        choices=['adam', 'sgd', 'rprop', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--scheduler', type=str, default=None, help='scheduler',
                        choices=['cosine', 'cosine_restart', 'lambda'])
    # construct cascade parameters
    parser.add_argument('--constr_casc', action='store_true', help='use constructive cascade network')
    parser.add_argument('--cascade_hidden_dim', type=int, default=64, help='hidden dimension of cascade layer')
    parser.add_argument('--cascade_dropout', type=float, default=0.0, help='dropout rate of cascade layer')
    parser.add_argument('--max_cascade_layers', type=int, default=None, help='maximum number of cascade layers')
    parser.add_argument('--threshold', type=float, default=0.03, help='threshold for adding a new cascade layer')
    parser.add_argument('--l2_ratio', type=float, default=0.2, help='ratio of l2 step size for cascade layer')
    parser.add_argument('--l3_ratio', type=float, default=0.1, help='ratio of l3 step size for cascade layer')
    # other parameters
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--logdir', type=str, default='./logs/', help='log directory')
    parser.add_argument('--log_epoch', type=int, default=25, help='interval of epochs of taking the log')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    main(args)
