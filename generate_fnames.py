import os
import os.path as osp
import pickle
import logging


def create_model_folder(args, model):
    if not osp.isdir(args.outpath):
        os.makedirs(args.outpath)

    model_fname = get_model_fname(args, model)
    outpath = osp.join(args.outpath, model_fname)

    if osp.isdir(outpath):
        logging.warn(f"Model output {outpath} already exists. Please delete it, rename it, or store it somewhere else so that the existing files are not overwritten.")
        # exit(1)
    else:
        os.makedirs(outpath)

    model_kwargs = {'model_name': model_fname, 'learning_rate': args.lr}

    with open(f'{outpath}/model_kwargs.pkl', 'wb') as f:
        pickle.dump(model_kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outpath


def get_model_fname(args, model):
    model_name = type(model).__name__
    model_fname = f"{model_name}_maxdim_{args.maxdim}_numBasisfn_{args.num_basis_fn}"
    if args.suffix is not None:
        model_fname += f"_{args.suffix}"
    return model_fname
