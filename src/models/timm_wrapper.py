import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as T


def set_normalization_mean_std(transforms, mean=[0.4425, 0.4380, 0.4332], std=[0.1858, 0.1812, 0.1795]):
    # this function sets the mean and std of the normalization transform to the values
    # calculate on vehicle-x training dataset
    for transform in transforms.transforms:
        if isinstance(transform, T.Normalize):
            transform.mean = mean
            transform.std = std


def get_model(model_name, num_classes, pretrained=True):
    # this function works as a wrapper for timm.create_model, which can be used to create a model
    # with a given name and number of classes. It also returns the train and test transforms
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    train_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model), is_training=True)
    test_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model), is_training=False)
    set_normalization_mean_std(train_transform)
    set_normalization_mean_std(test_transform)
    return model, train_transform, test_transform
