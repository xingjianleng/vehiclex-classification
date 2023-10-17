def str2bool(v):
    # used for argparse to parse boolean
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected')


def float_or_none(value):
    # used for argparse to parse float or None
    if value.lower() == 'none':
        return None
    return float(value)
