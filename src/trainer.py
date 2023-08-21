import time

import torch
import torch.nn.functional as F


class NetworkTrainer(object):
    def __init__(self, model, logdir, args):
        self.model = model
        self.args = args
        self.logdir = logdir

    def train(self, epoch, dataloader, optimizer, scheduler, log_interval):
        # set model to training mode
        self.model.train()

        # initialize loss and accuracy
        losses = 0.
        correct = 0

        t0 = time.time()

        # iterate over training batches
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            out = self.model(data)
            if self.args.loss == 'ce':
                loss = F.cross_entropy(out, target)
            # TODO: Implement focal loss
            # elif self.args.loss == 'focal':
            #     loss = focal_loss(out, target)
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

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))

            # print training status
            if (batch_idx + 1) % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}, '
                      f'Acc: {correct / (batch_idx + 1):.6f}')
                
        # print training time
        t1 = time.time()
        print(f'Training time: {t1 - t0:.2f}s')

    def test():
        # TODO: Implement test function
        pass
