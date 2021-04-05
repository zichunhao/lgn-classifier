import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(1, 'data_processing/')
sys.path.insert(1, 'lgn/')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import args
from args import setup_argparse

from data_processing.make_pytorch_data import initialize_data
from lgn.models.lgn_jet_classifier import LGNJetClassifier
from train import train_loop
from generate_fnames import create_model_folder

import json
import pickle

from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import math
from math import inf

import logging

if __name__ == "__main__":
    args = setup_argparse()

    if args.logging:
        logging.basicConfig(level=logging.INFO)

    with open("args_cache.json", "w") as f:
        json.dump(vars(args), f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Num of GPUs: {torch.cuda.device_count()}")

    if device.type == 'cuda':
        print(f"GPU tagger: {torch.cuda.current_device()}")
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    if args.dtype == 'float64':
        dtype = torch.float64
    elif args.dtype == 'float':
        dtype = torch.float
    elif args.dtype == 'double':
        dtype = torch.double
    else:
        raise NotImplementedError(f"Data type {args.dtype} is not implemented. Please choose 'float', 'float64', or 'double'.")

    print(f'Working on {str(device).upper()} with {args.dtype}')

    # Initialize and load dataset
    train_loader, test_loader, valid_loader = initialize_data(args.file_path, args.batch_size, args.num_train, args.num_test, args.num_val)

    # Initialize model
    model = LGNJetClassifier(maxdim=args.maxdim, max_zf=args.max_zf,
                             num_cg_levels=args.num_cg_levels,  num_channels=args.num_channels,
                             weight_init=args.weight_init, level_gain=args.level_gain,
                             num_basis_fn=args.num_basis_fn, output_layer=args.output_layer,
                             num_mpnn_layers=args.num_mpnn_layers, activation=args.activation,
                             p4_into_CG=args.p4_into_CG, add_beams=args.add_beams, scale=1.,
                             full_scalars=args.full_scalars, mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                             device=device, dtype=dtype)

    if (next(model.parameters()).is_cuda):
        print('The model is initialized on GPU..')
    else:
        print('The model is initialized on CPU..')

    # training
    if args.train:
        # load existing model
        if args.load_to_train:
            outpath = args.outpath + args.load_model_path
            model.load_state_dict(torch.load(f'{outpath}/epoch_{args.load_epoch}_weights.pth'))
        # create new model
        else:
            outpath = create_model_folder(args, model)

        optimizer = torch.optim.Adam(model.parameters(), args.lr_init)
        train_loop(args, model=model, optimizer=optimizer, outpath=outpath, train_loader=train_loader, valid_loader=valid_loader, device=device)
