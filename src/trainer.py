import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn.functional as F


class NetworkTrainer(object):
    # a wrapper class for training and testing a model
    def __init__(self, model, logdir, writer, args):
        self.model = model
        self.args = args
        self.writer = writer
        self.logdir = logdir

    def train(self, epoch, dataloader, optimizer, scheduler):
        # set model to training mode
        self.model.train()

        # initialize loss and accuracy
        losses = 0.
        correct = 0
        preds = []
        tgts = []

        t0 = time.time()

        # iterate over training batches
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            out = self.model(data)
            loss = F.cross_entropy(out, target)
        
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss and accuracy
            losses += loss.item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            preds.append(pred.detach().cpu().numpy())
            tgts.append(target.detach().cpu().numpy())

            # step scheduler to update learning rate if we are using one
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))

            # logging
            if (batch_idx + 1) % self.args.log_interval == 0 or batch_idx + 1 == len(dataloader):
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train epoch: {epoch}, batch:{(batch_idx + 1)}, '
                      f'loss: {losses / (batch_idx + 1):.3f}, time: {t_epoch:.1f}')

        # calculate average loss and accuracy over all training batches
        train_loss = losses / len(dataloader)
        train_acc = correct / len(dataloader.dataset) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(np.concatenate(tgts), np.concatenate(preds), average='macro',
                                                                    zero_division=0)
        precision *= 100
        recall *= 100
        f1 *= 100    
    
        # log training results to tensorboard
        self.writer.add_scalar("charts/train_loss", train_loss, epoch)
        self.writer.add_scalar("charts/train_acc", train_acc, epoch)
        self.writer.add_scalar("charts/train_precision", precision, epoch)
        self.writer.add_scalar("charts/train_recall", recall, epoch)
        self.writer.add_scalar("charts/train_f1", f1, epoch)
        self.writer.add_scalar("charts/lr", optimizer.param_groups[0]['lr'], epoch)

        # print training time and metrics on training set
        t1 = time.time()
        print(f'Train loss: {train_loss:.6f}, Accuracy: {train_acc:.4f}%, '
              f'precision: {precision:.4f}%, recall: {recall:.4f}%, f1: {f1:.4f}%')
        print(f'Training time: {t1 - t0:.2f}s')
        return train_loss, train_acc

    def test(self, dataloader, epoch=None):
        # set model to evaluation mode
        self.model.eval()

        # initialize loss and accuracy
        test_loss = 0.
        correct = 0
        preds = []
        tgts = []

        # disable gradient computation
        with torch.no_grad():
            # iterate over test batches
            for data, target in dataloader:
                data, target = data.cuda(), target.cuda()
                out = self.model(data)
                loss = F.cross_entropy(out, target)
        
                # accumulate loss and accuracy
                test_loss += loss.item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                preds.append(pred.detach().cpu().numpy())
                tgts.append(target.detach().cpu().numpy())

        # calculate average loss and accuracy over all test batches
        test_loss /= len(dataloader)
        test_acc = correct / len(dataloader.dataset) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(np.concatenate(tgts), np.concatenate(preds), average='macro',
                                                                   zero_division=0)
        precision *= 100
        recall *= 100
        f1 *= 100

        # if epoch is not None, then this is a validation test
        if epoch:
            self.writer.add_scalar("charts/val_loss", test_loss / len(dataloader), epoch)
            self.writer.add_scalar("charts/val_acc", test_acc, epoch)
            self.writer.add_scalar("charts/val_precision", precision, epoch)
            self.writer.add_scalar("charts/val_recall", recall, epoch)
            self.writer.add_scalar("charts/val_f1", f1, epoch)

        # print test results
        print(f'\n{"Val" if epoch else "Test"} set: Average loss: {test_loss:.6f}, Accuracy: {test_acc:.4f}%, '
                f'Precision: {precision:.4f}%, Recall: {recall:.4f}%, F1: {f1:.4f}%\n')
        return test_loss, test_acc, precision, recall, f1
