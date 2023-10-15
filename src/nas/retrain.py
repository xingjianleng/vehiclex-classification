import json

from nni.nas.evaluator.pytorch import Classification
from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.nas.nn.pytorch import MutableLinear
from nni.nas.space import model_context

from src.utils.nas_optim import get_optim


def make_model(args):
    with open(args.arch_path, 'r') as fp:
        exported_arch = json.load(fp)

    with model_context(exported_arch):
        model = DartsSpace(
            width=args.retrain_width,
            num_cells=args.retrain_num_cells,
            dataset='imagenet'
        )

    model.num_labels = args.num_classes
    model.classifier = MutableLinear(model.classifier.in_features, model.num_labels)
    return model


def retrain_model(train_loader, valid_loader, args):
    model = make_model(args)
    
    optimizer = get_optim(args)

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
