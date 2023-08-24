import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn.functional as F

from src.loss.focal_loss import focal_loss


class NetworkTrainer(object):
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
            if self.args.loss == 'ce':
                loss = F.cross_entropy(out, target)
            elif self.args.loss == 'focal':
                loss = focal_loss(out, target, gamma=self.args.gamma)
            else:
                raise NotImplementedError(f'Unknown loss function: {self.args.loss}')
        
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

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))

        train_loss = losses / len(dataloader)
        train_acc = correct / len(dataloader.dataset) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(np.concatenate(tgts), np.concatenate(preds), average='macro',
                                                                    zero_division=0)
        precision *= 100
        recall *= 100
        f1 *= 100    
    
        self.writer.add_scalar("charts/train_loss", train_loss, epoch)
        self.writer.add_scalar("charts/train_acc", train_acc, epoch)
        self.writer.add_scalar("charts/train_precision", precision, epoch)
        self.writer.add_scalar("charts/train_recall", recall, epoch)
        self.writer.add_scalar("charts/train_f1", f1, epoch)
        self.writer.add_scalar("charts/lr", optimizer.param_groups[0]['lr'], epoch)

        # print training time
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
                if self.args.loss == 'ce':
                    loss = F.cross_entropy(out, target)
                elif self.args.loss == 'focal':
                    loss = focal_loss(out, target, gamma=self.args.gamma)
                else:
                    raise NotImplementedError(f'Unknown loss function: {self.args.loss}')
        
                # accumulate loss and accuracy
                test_loss += loss.item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                preds.append(pred.detach().cpu().numpy())
                tgts.append(target.detach().cpu().numpy())

        test_loss /= len(dataloader)
        test_acc = correct / len(dataloader.dataset) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(np.concatenate(tgts), np.concatenate(preds), average='macro',
                                                                   zero_division=0)
        precision *= 100
        recall *= 100
        f1 *= 100

        if epoch:
            self.writer.add_scalar("charts/val_loss", test_loss / len(dataloader), epoch)
            self.writer.add_scalar("charts/val_acc", test_acc, epoch)
            self.writer.add_scalar("charts/val_precision", precision, epoch)
            self.writer.add_scalar("charts/val_recall", recall, epoch)
            self.writer.add_scalar("charts/val_f1", f1, epoch)

        # print test results
        
        print(f'\n{"Val" if epoch else "Test"} set: Average loss: {test_loss:.6f}, Accuracy: {test_acc:.4f}%, '
                f'Precision: {precision:.4f}%, Recall: {recall:.4f}%, F1: {f1:.4f}%\n')
        return test_loss, test_acc
