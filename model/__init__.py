import os
import sys
from model.waft_v1 import ViTWarpV8
from model.waft_v2 import WAFTv2

def fetch_model(args):
    if args.algorithm == 'waftv1':
        model = ViTWarpV8(args)
    elif args.algorithm == 'waftv2':
        model = WAFTv2(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model