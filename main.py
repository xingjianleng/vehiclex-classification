import argparse
import datetime
from distutils.dir_util import copy_tree
import json
import os
import shutil
import sys

import numpy as np
import nni
from nni.nas.evaluator.pytorch import DataLoader as DataLoaderNNI
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import tqdm

from src.trainer import NetworkTrainer
from src.datasets.vehicle_x import Vehicle_X
from src.models.timm_wrapper import get_model
from src.nas.search import darts_search
from src.nas.retrain import retrain_model
from src.utils.argparse_type import float_or_none
from src.utils.draw_curve import draw_curve
from src.utils.eval import evaluate_model
from src.utils.logger import Logger


def prepare_nni_dataloader(base, bs, num_workers):
    # transforms
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250]),
    ])
    test_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250]),
    ])

    # datasets and dataloaders
    train_data = nni.trace(Vehicle_X)(
        root=os.path.expanduser(os.path.join(base, 'train')), transform=train_transform)
    val_data = nni.trace(Vehicle_X)(
        root=os.path.expanduser(os.path.join(base, 'val')), transform=test_transform)
    test_data = nni.trace(Vehicle_X)(
        root=os.path.expanduser(os.path.join(base, 'test')), transform=test_transform)

    train_loader = DataLoaderNNI(
        train_data, batch_size=bs, num_workers=num_workers
    )

    val_loader = DataLoaderNNI(
        val_data, batch_size=bs, num_workers=num_workers
    )

    test_loader = DataLoaderNNI(
        test_data, batch_size=bs, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def copy_script(logdir):
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


def nas_search_main(args):
    if args.eval:
        raise NotImplementedError('Evaluation mode not implemented for NAS search')
    
    logdir = f'{args.logdir}{"DEBUG_" if is_debug else ""}NASsearch_' \
             f'lr{args.search_lr}_b{args.tr_batch_size}_e{args.search_epochs}_' \
             f'optim{args.search_optim}_width{args.search_width}' \
             f'_cell{args.search_num_cells}_wd{args.search_weight_decay}_' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
    copy_script(logdir)

    train_loader, val_loader, _ = prepare_nni_dataloader(args.base, args.search_batch_size, args.num_workers)

    print(f'Start searching...')
    exported_arch = darts_search(train_loader, val_loader, args)
    with open(os.path.join(logdir, 'exported_arch.json'), 'w') as fp:
        json.dump(exported_arch, fp)
    print(f'Searching finished...')


def nas_training_main(args):
    if args.eval:
        raise NotImplementedError('Evaluation mode not implemented for NAS retrain')

    train_loader, val_loader, test_loader = prepare_nni_dataloader(args.base, args.retrain_batch_size, args.num_workers)

    logdir = f'{args.logdir}{"DEBUG_" if is_debug else ""}NASretrain_' \
             f'lr{args.retrain_lr}_b{args.tr_batch_size}_e{args.retrain_epochs}_' \
             f'optim{args.retrain_optim}_width{args.retrain_width}' \
             f'_cell{args.retrain_num_cells}_wd{args.retrain_weight_decay}_' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
    copy_script(logdir)

    model = retrain_model(train_loader, val_loader, args)
    model.cuda()

    test_results = evaluate_model(model, test_loader)
    with open(os.path.join(logdir, 'test_results.txt'), 'w') as f:
        for test_result in test_results:
            f.write(f'{test_result:.6f}\n')


def baseline_main(args):
    # get models and transformations from timm, and transfer to GPU
    model, train_transform, test_transform = get_model(args.model, args.num_classes, pretrained=args.pretrained)
    model.cuda()

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
    logdir = f'{args.logdir}{"DEBUG_" if is_debug else ""}BASELINE_' \
             f'lr{args.lr}_b{args.tr_batch_size}_e{args.epochs}_' \
             f'optim{args.optim}_model{args.model}_scheduler{args.scheduler}' \
             f'_wd{args.weight_decay}_seed{args.seed}_pretrained{args.pretrained}_' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}' if not args.eval \
        else f'{args.logdir}{args.dataset}/EVAL_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
    copy_script(logdir)

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
    # training parameter
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=150, help='number of train epochs')
    # optimizer parameter
    parser.add_argument('--optim', type=str, default='adam', help='optimizer',
                        choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--scheduler', action='store_true', help='enable lambda scheduler')
    # nas search parameters
    parser.add_argument('--nas_search', action='store_true', help='use neural architecture search (NAS)')
    parser.add_argument('--search_batch_size', type=int, default=64, help='batch size for search')
    parser.add_argument('--search_epochs', type=int, default=100, help='number of search epochs')
    parser.add_argument('--search_lr', type=float, default=1e-3, help='learning rate for search')
    parser.add_argument('--search_weight_decay', type=float, default=1e-4, help='weight decay for search')
    parser.add_argument('--search_width', type=int, default=16, help='width of the network')
    parser.add_argument('--search_num_cells', type=int, default=8, help='number of cells in the network')
    parser.add_argument('--search_optim', type=str, default='adam', help='optimizer for search',
                        choices=['adam', 'sgd'])
    parser.add_argument('--search_grad_clip', type=float_or_none, default=5., help='gradient clipping')
    # nas training parameters
    parser.add_argument('--nas_retrain', action='store_true', help='retrain NAS model')
    parser.add_argument('--arch_path', type=str, default='./exported_arch.json', help='path to exported architecture')
    parser.add_argument('--retrain_batch_size', type=int, default=64, help='batch size for retrain')
    parser.add_argument('--retrain_width', type=int, default=32, help='width of the network')
    parser.add_argument('--retrain_num_cells', type=int, default=20, help='number of cells in the network')
    parser.add_argument('--retrain_lr', type=float, default=5e-3, help='learning rate for retrain')
    parser.add_argument('--retrain_weight_decay', type=float, default=5e-4, help='weight decay for retrain')
    parser.add_argument('--retrain_epochs', type=int, default=200, help='number of retrain epochs')
    parser.add_argument('--retrain_optim', type=str, default='adam', help='optimizer for retrain',
                        choices=['adam', 'sgd'])
    # other parameters
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--logdir', type=str, default='./logs/', help='log directory')
    parser.add_argument('--log_interval', type=int, default=200, help='interval of epochs of taking the log')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()

    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('No sys.gettrace')
        is_debug = False

    if args.nas_search:
        nas_search_main(args)
    elif args.nas_retrain:
        nas_training_main(args)
    else:
        baseline_main(args)
