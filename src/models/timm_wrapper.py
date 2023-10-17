import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def get_model(model_name, num_classes, pretrained=True):
    # this function works as a wrapper for timm.create_model, which can be used to create a model
    # with a given name and number of classes. It also returns the train and test transforms
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    train_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model), is_training=True)
    test_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model), is_training=False)
    return model, train_transform, test_transform
