"""SambaNova boilerplate main method."""

import argparse
import sys
from typing import Tuple


import torch
import torch.nn as nn
import torchvision

from sambaflow import samba

import sambaflow.samba.utils as utils
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.pef_utils import get_pefmeta
from sambaflow.samba.utils.dataset.mnist import dataset_transform

from sn_model_args import *
from sn_model_model import *
from sn_model_other import *

def main(argv):
    """Run main code."""
    utils.set_seed(256)
    args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)

    ipt, tgt = Model.get_fake_inputs(args)
    """
    print(f"args.batch_size: {args.batch_size}")
    print(f"args.num_features: {args.num_features}")
    print(f"args.num_classes: {args.num_classes}")
    """
    model = Model()

    samba.from_torch_(model)

    inputs = (ipt, tgt)

    # Instantiate an optimizer.
    if args.inference:
        optimizer = None
    else:
        optimizer = samba.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.command == "compile":
        # Run model analysis and compile, this step will produce a PEF.
        samba.session.compile(model,
                              inputs,
                              optimizer,
                              name='model_1',
                              app_dir=utils.get_file_dir(__file__),
                              config_dict=vars(args),
                              pef_metadata=get_pefmeta(args, model))
    elif args.command == "test":
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        outputs = model.output_tensors
        test(args, model, inputs, outputs)
    elif args.command == "run":
        utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
        train(args, model, optimizer)


if __name__ == '__main__':
    main(sys.argv[1:])
