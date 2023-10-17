import argparse

import thop
import timm
import torch

from src.nas.retrain import make_model


def main(args):
    # this script is used to compute the FLOPs and params of the model
    if args.is_timm:
        # evaluate the model from timm
        model = timm.create_model(args.model, num_classes=args.num_classes, pretrained=False)
        print(f'{args.model} from timm')
    else:
        # evaluate the model from neural architecture search results
        model = make_model(args)
        print('model from exported architecture')
    # compute the FLOPs and params
    img = torch.randn(1, 3, 224, 224)
    flops, params = thop.profile(model, inputs=(img, ))
    flops, params = thop.clever_format([flops, params], '%.3f')
    print(f'FLOPs: {flops}, params: {params}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_timm', action='store_true', help='whether the model is in timm')
    parser.add_argument('--model', type=str, default='resnet50', help='the name of the model from timm')
    parser.add_argument('--num_classes', type=int, default=1362, help='number of classes in the dataset')
    parser.add_argument('--arch_path', type=str, default='./exported_arch.json', help='path to exported architecture')
    parser.add_argument('--retrain_width', type=int, default=32, help='width of the network')
    parser.add_argument('--retrain_num_cells', type=int, default=20, help='number of cells in the network')
    args = parser.parse_args()

    main(args)
