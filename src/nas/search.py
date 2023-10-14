import torch.optim as optim
from nni.nas.evaluator.pytorch import Classification
from nni.nas.experiment import NasExperiment
from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.nas.nn.pytorch import MutableLinear
from nni.nas.strategy import DARTS as DartsStrategy

from src.utils.nas_optim import get_optim


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

    optimizer = get_optim(args)

    evaluator = Classification(
        learning_rate=args.search_lr,
        weight_decay=args.search_weight_decay,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        max_epochs=args.search_epochs,
        num_classes=args.num_classes,
        optimizer=optimizer,
        export_onnx=False
    )

    strategy = DartsStrategy(gradient_clip_val=args.search_grad_clip)

    experiment = NasExperiment(model_space, evaluator, strategy)
    experiment.run()

    exported_arch = experiment.export_top_models(formatter='dict')[0]
    return exported_arch
