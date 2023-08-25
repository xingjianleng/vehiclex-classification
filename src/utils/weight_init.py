import torch.nn.init as init


def layer_init(layer, mode='xavier'):
    if mode == 'xavier':
        init.xavier_normal_(layer.weight)
        init.zeros_(layer.bias)
    elif mode == 'kaiming':
        init.kaiming_normal_(layer.weight)
        init.zeros_(layer.bias)
    else:
        raise NotImplementedError('other initialization methods are not implemented')
    return layer
