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
    print(args)
    if args.logging:
        logging.basicConfic(level=logging.INFO)

    with open("args_cache.json", "w") as f:
        json.dump(vars(args), f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Num of GPUs: {torch.cuda.device_count()}")

    if device.type == 'cuda':
        print(f"GPU tagger: {torch.cuda.current_device()}")
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
