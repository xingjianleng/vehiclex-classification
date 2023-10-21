import json

from nni.nas.evaluator.pytorch import Classification
from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.nas.nn.pytorch import MutableLinear
from nni.nas.space import model_context
from timm.models._efficientnet_builder import _init_weight_goog

from src.utils.nas_optim import get_optim


def make_model(args):
    # construct model from exported arch, which is the searched cell architecture and should be a json file
    with open(args.arch_path, 'r') as fp:
        exported_arch = json.load(fp)

    with model_context(exported_arch):
        model = DartsSpace(
            width=args.retrain_width,
            num_cells=args.retrain_num_cells,
            dataset='imagenet'
        )

    # NOTE: nni library only supports 1000 classes for imagenet
    #       we manually change the number of classes here
    model.num_labels = args.num_classes
    model.classifier = MutableLinear(model.classifier.in_features, model.num_labels)

    # model initialization: use google initialization
    for n, m in model.named_modules():
        _init_weight_goog(m, n)

    return model


def retrain_model(train_loader, valid_loader, args):
    # retrain the model with the searched cell architecture using pytorch-lightning trainer
    model = make_model(args)
    
    # decide the optimizer to use
    optimizer = get_optim(args)

    # initialize the classification task evaluator and fit the model
    evaluator = Classification(
        learning_rate=args.retrain_lr,
        weight_decay=args.retrain_weight_decay,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        max_epochs=args.retrain_epochs,
        num_classes=args.num_classes,
        optimizer=optimizer,
        export_onnx=False
    )

    evaluator.fit(model)

    return model
