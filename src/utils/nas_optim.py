import torch.optim as optim


def get_optim(args):
    # this function returns the optimizer given the arguments, used for NAS search
    if args.search_optim == 'adam':
        optimizer = optim.Adam
    elif args.search_optim == 'sgd':
        optimizer = optim.SGD
    else:
        raise NotImplementedError(f'Optimizer {args.search_optim} not implemented')
    
    return optimizer
