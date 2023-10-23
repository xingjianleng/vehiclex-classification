from nni.nas.evaluator.pytorch import ClassificationModule, Lightning, Trainer
from nni.nas.experiment import NasExperiment
from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.nas.nn.pytorch import MutableLinear
from nni.nas.strategy import DARTS as DartsStrategy
import torch.optim as optim

from src.utils.nas_optim import get_optim


class DartsClassificationModule(ClassificationModule):
    """
    Customize the DARTS model to use the PyTorch Lightning interface. Use Adam optimizer with CosineAnnealingLR
    Modified from: https://nni.readthedocs.io/en/latest/tutorials/darts.html#id2
    """
    def __init__(
        self,
        learning_rate: float=1e-3,
        weight_decay: float=5e-4,
        num_classes=1362,
        max_epochs=100,
        optimizer=optim.Adam
    ):
        # Training length will be used in LR scheduler
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False, num_classes=num_classes)

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-8)
        }

    def training_step(self, batch, batch_idx):
        """Training step, customized with auxiliary loss."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        # Set drop path probability before every epoch. This has no effect if drop path is not enabled in model.
        self.model.set_drop_path_prob(self.model.drop_path_prob * self.current_epoch / self.max_epochs)

        # Logging learning rate at the beginning of every epoch
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True)


def darts_search(train_loader, valid_loader, args):
    # imagenet gives more downsampling in the first stage
    model_space = DartsSpace(
        width=args.search_width,
        num_cells=args.search_num_cells,
        dataset='imagenet'
    )
    # NOTE: nni library only supports 1000 classes for imagenet
    #       we manually change the number of classes here
    model_space.num_labels = args.num_classes
    model_space.classifier = MutableLinear(model_space.classifier.in_features, model_space.num_labels)

    # decide the optimizer to use
    optimizer = get_optim(args)

    # initialize the classification task evaluator and fit the model
    evaluator = Lightning(
        DartsClassificationModule(args.search_lr, args.search_weight_decay, args.num_classes, args.search_epochs, optimizer),
        Trainer(max_epochs=args.search_epochs),
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    # we use the default strategy for DARTS, but we enable gradient clipping to stabilize training
    strategy = DartsStrategy(gradient_clip_val=args.search_grad_clip)

    # initialize the NAS experiment and run it
    experiment = NasExperiment(model_space, evaluator, strategy)
    experiment.run()

    # return the only the best architecture found
    exported_arch = experiment.export_top_models(formatter='dict')[0]
    return exported_arch
