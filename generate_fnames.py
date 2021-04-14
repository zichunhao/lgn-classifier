import os
import os.path as osp
import pickle

def create_model_folder(args, model):
    if not osp.isdir(args.outpath):
        os.makedirs(args.outpath)

    model_fname = get_model_fname(args, model)
    outpath = osp.join(args.outpath, model_fname)

    if osp.isdir(outpath):
        print(f"Model output {outpath} already exists. Please delete it, rename it, or store it somewhere else so that the existing files are not overwritten.")
        exit(1)
    else:
        os.makedirs(outpath)

    model_kwargs = {'model_name': model_fname, 'learning_rate': args.lr}

    with open(f'{outpath}/model_kwargs.pkl', 'wb') as f:
        pickle.dump(model_kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outpath

def get_model_fname(args, model):
    model_name = type(model).__name__
    model_fname = f"{model_name}_numEpochs={args.num_epochs}_batchSize={args.batch_size}_numTrain={args.num_train}_lr={args.lr}"
    return model_fname
