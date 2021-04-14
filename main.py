import torch
import matplotlib.pyplot as plt
import args
from args import setup_argparse

import pickle
import os
import os.path as osp
import sys
sys.path.insert(1, 'data_processing/')
sys.path.insert(1, 'lgn/')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from data_processing.make_pytorch_data import initialize_data
from lgn.models.lgn_jet_classifier import LGNJetClassifier
from lgn.models.autotest import lgn_tests
from train import train_loop
from generate_fnames import create_model_folder

import json
import logging

if __name__ == "__main__":
    args = setup_argparse()

    if args.print_logging:
        logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Num of GPUs: {torch.cuda.device_count()}")

    if device.type == 'cuda':
        print(f"GPU tagger: {torch.cuda.current_device()}")
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    if args.dtype == 'float64' or 'double':
        dtype = torch.float64
    elif args.dtype == 'float':
        dtype = torch.float
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
                             num_mpnn_layers=args.num_mpnn_layers, num_classes=args.num_classes, activation=args.activation,
                             p4_into_CG=args.p4_into_CG, add_beams=args.add_beams, scale=1.,
                             full_scalars=args.full_scalars, mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                             device=device, dtype=dtype)

    if (next(model.parameters()).is_cuda):
        print('The model is initialized on GPU...')
    else:
        print('The model is initialized on CPU...')

    # training
    # load existing model
    if args.load_to_train:
        outpath = args.outpath + args.load_model_path
        model.load_state_dict(torch.load(f'{outpath}/epoch_{args.load_epoch}_weights.pth'))
    # create new model
    else:
        outpath = create_model_folder(args, model)

    with open(f"{outpath}/args_cache.json", "w") as f:
        json.dump(vars(args), f)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    train_loop(args, model=model, optimizer=optimizer, outpath=outpath, train_loader=train_loader, valid_loader=valid_loader, device=device)

    # Test equivariance of models
    if args.test_equivariance:
        PATH_equivariance_results = f"{outpath}/model_evaluations/equivariance_test"
        if not osp.isdir(PATH_equivariance_results):
            os.makedirs(PATH_equivariance_results)
        # Test the equivariance of the model in the last epoch only
        if not args.test_over_all_epochs:
            # Load model
            PATH = f"{outpath}/epoch_{args.num_epochs}_weights.pth"
            model.load_state_dict(torch.load(PATH, map_location=device))
            lgn_test_results = lgn_tests(model, test_loader, args, args.num_epochs, cg_dict=model.cg_dict)

            # Saving test results
            with open(f"{PATH_equivariance_results}/equivariance_test_epoch_{args.num_epochs}.pkl", 'wb') as f:
                pickle.dump(lgn_test_results, f)

            # Plotting test results
            # Boost
            if 'boost_equivariance' in lgn_test_results.keys():
                boost_results = lgn_test_results['boost_equivariance']
                # Original format: [[gamma1, dev1], [gamma2, dev2], ...]
                gamma = [boost_results[i][0] for i in range(len(boost_results))]
                boost_dev = [boost_results[i][1] for i in range(len(boost_results))]
                # plotting
                fig, ax = plt.subplots()
                fig, ax = plt.subplots()
                ax.plot(gamma, boost_dev, label='boost')
                ax.set_xlabel(r"Boost factor $\gamma$")
                ax.set_ylabel('Relative deviation')
                ax.set_title('Relative deviations for boost equivariance test', y=1.05)
                plt.savefig(f'{PATH_equivariance_results}/boost_equivariance_test_epoch_{args.num_epochs}.{args.fig_format}')
                plt.close(fig)
            # Rotation
            if 'rotation_equivariance' in lgn_test_results.keys():
                rot_results = lgn_test_results['rotation_equivariance']
                # Original format: [[angle1, dev1], [angle2, dev2], ...]
                angle = [rot_results[i][0] for i in range(len(rot_results))]
                rot_dev = [rot_results[i][1] for i in range(len(rot_results))]
                # plotting
                fig, ax = plt.subplots()
                ax.plot(angle, rot_dev, label='rotation')
                ax.set_xlabel(r"Rotation angle")
                ax.set_ylabel('Relative deviation')
                ax.set_title('Relative deviations for rotational equivariance test', y=1.05)
                plt.savefig(f'{PATH_equivariance_results}/rot_equivariance_test_epoch_{args.num_epochs}.{args.fig_format}')
                plt.close(fig)

        # Test the equivariance over all epochs
        else:
            for epoch in range(args.num_epochs):
                # Load model
                PATH = f"{outpath}/epoch_{epoch+1}_weights.pth"
                model.load_state_dict(torch.load(PATH, map_location=device))
                print(f"Testing equivariance for epoch {epoch}...")
                lgn_test_results = lgn_tests(model, test_loader, args, epoch+1, cg_dict=model.cg_dict)

                # Saving test results
                with open(f"{PATH_equivariance_results}/equivariance_test_epoch_{epoch+1}.pkl", 'wb') as f:
                    pickle.dump(lgn_test_results, f)

                # Plotting test results
                # Boost
                if 'boost_equivariance' in lgn_test_results.keys():
                    boost_results = lgn_test_results['boost_equivariance']
                    # Original format: [[gamma1, dev1], [gamma2, dev2], ...]
                    gamma = [boost_results[i][0] for i in range(len(boost_results))]
                    boost_dev = [boost_results[i][1] for i in range(len(boost_results))]
                    # plotting
                    fig, ax = plt.subplots()
                    fig, ax = plt.subplots()
                    ax.plot(gamma, boost_dev, label='boost')
                    ax.set_xlabel(r"Boost factor $\gamma$")
                    ax.set_ylabel('Relative deviation')
                    ax.set_title(f'Relative deviations for boost equivariance test at epoch {epoch+1}', y=1.05)
                    plt.savefig(f'{PATH_equivariance_results}/boost_equivariance_test_epoch_{epoch+1}.{args.fig_format}')
                    plt.close(fig)
                # Rotation
                if 'rotation_equivariance' in lgn_test_results.keys():
                    rot_results = lgn_test_results['rotation_equivariance']
                    # Original format: [[angle1, dev1], [angle2, dev2], ...]
                    angle = [rot_results[i][0] for i in range(len(rot_results))]
                    rot_dev = [rot_results[i][1] for i in range(len(rot_results))]
                    # plotting
                    fig, ax = plt.subplots()
                    ax.plot(angle, rot_dev, label='rotation')
                    ax.set_xlabel(r"Rotation angle")
                    ax.set_ylabel('Relative deviation')
                    ax.set_title(f'Relative deviations for rotational equivariance test at epoch {epoch + 1}', y=1.05)
                    plt.savefig(f'{PATH_equivariance_results}/rot_equivariance_test_epoch_{epoch+1}.{args.fig_format}')
                    plt.close(fig)
