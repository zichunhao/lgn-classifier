import torch
import torch.nn as nn
from torch.nn.functional import softmax
import sys
sys.path.insert(1, 'data_processing/')
sys.path.insert(1, 'lgn/')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

import pickle
import os
import os.path as osp
import time

from plot_results import plot_confusion_matrix, plot_roc_curve

def train(args, model, loader, epoch, outpath, is_train=True, optimizer=None, lr=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_train:
        model.train()
        assert (optimizer is not None), "Please specify the optimizer for the training loop."
        assert (lr is not None), "Please specify the learning rate (lr) for the training loop."
    else:
        model.eval()

    losses_per_epoch = []
    avg_loss_per_epoch = []
    correct_preds = 0
    accuracy = 0
    # For confusion matrices
    predictions = []
    targets = []
    # For ROC curves
    predictions_onehot = []
    targets_onehot = []

    for i, batch in enumerate(loader):
        t0 = time.time()

        X = batch
        Y = batch['jet_types'].to(device)

        # Forward propagation
        preds = model(X)

        # Backward propagation
        loss = nn.CrossEntropyLoss()
        batch_loss = loss(preds, Y.long().argmax(axis=1))

        if is_train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        t1 = time.time()

        # Correction prediction
        correct_preds += (preds.argmax(axis=1) == Y.argmax(axis=1)).sum().item()
        accuracy = correct_preds / (args.batch_size*len(loader))

        # if is_train:
        #     print(f"batch {i+1}/{len(loader)}, train_loss={batch_loss.item()}, train_accuracy={accuracy}, time_duration={t1-t0}", end='\r', flush=True)
        # else:
        #     print(f"batch {i+1}/{len(loader)}, valid_loss={batch_loss.item()}, valid_accuracy={accuracy}, time_duration={t1-t0}", end='\r', flush=True)

        losses_per_epoch.append(batch_loss.item())

        # For confusion matrix
        predictions.append(preds.detach().cpu().numpy())
        targets.append(Y.detach().cpu().numpy())

        # For ROC curves
        preds_prob = softmax(preds, dim=1)  # Normalize the total probability for a jet
        predictions_onehot.append(preds_prob.detach().cpu().numpy())
        targets_onehot.append(Y.detach().cpu().numpy())

    avg_loss_per_epoch = sum(losses_per_epoch)/len(losses_per_epoch)

    # Confusion matrices
    predictions = np.concatenate(predictions).argmax(axis=1)
    targets = np.concatenate(targets).argmax(axis=1)
    confusion_matrix = sklearn.metrics.confusion_matrix(targets, predictions, normalize='true')

    if is_train:
        plot_confusion_matrix(args, confusion_matrix, epoch, outpath, is_train=True)
    else:
        plot_confusion_matrix(args, confusion_matrix, epoch, outpath, is_train=False)

    # ROC curves
    targets_onehot = np.concatenate(targets_onehot)
    predictions_onehot = np.concatenate(predictions_onehot)
    if is_train:
        tpr, fpr, auc = plot_roc_curve(args, predictions_onehot, targets_onehot, epoch, outpath, is_train=True)
        PATH_roc_pkl = f"{outpath}/model_evaluations/roc_curves/pkl_files"
        if not osp.isdir(PATH_roc_pkl):
            os.makedirs(PATH_roc_pkl)
        with open(f"{PATH_roc_pkl}/tpr_train_epoch_{epoch+1}.pkl", 'wb') as f:
            pickle.dump(tpr, f)
        with open(f"{PATH_roc_pkl}/fpr_train_epoch_{epoch+1}.pkl", 'wb') as f:
            pickle.dump(fpr, f)
        with open(f"{PATH_roc_pkl}/auc_train_epoch_{epoch+1}.pkl", 'wb') as f:
            pickle.dump(auc, f)

    else:
        tpr, fpr, auc = plot_roc_curve(args, predictions_onehot, targets_onehot, epoch, outpath, is_train=False)
        PATH_roc_pkl = f"{outpath}/model_evaluations/roc_curves/pkl_files"
        if not osp.isdir(PATH_roc_pkl):
            os.makedirs(PATH_roc_pkl)
        with open(f"{PATH_roc_pkl}/tpr_valid_epoch_{epoch+1}.pkl", 'wb') as f:
            pickle.dump(tpr, f)
        with open(f"{PATH_roc_pkl}/fpr_valid_epoch_{epoch+1}.pkl", 'wb') as f:
            pickle.dump(fpr, f)
        with open(f"{PATH_roc_pkl}/auc_valid_epoch_{epoch+1}.pkl", 'wb') as f:
            pickle.dump(auc, f)

    return avg_loss_per_epoch, accuracy

@torch.no_grad()
def test(args, model, test_loader, epoch, outpath, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        test_pred, acc = train(args, model, test_loader, epoch, outpath, is_train=False, optimizer=None, lr=None, device=device)
    return test_pred, acc

def train_loop(args, model, optimizer, outpath, train_loader, valid_loader, device):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t0_initial = time.time()

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    dts = []

    best_valid_loss = math.inf
    stale_epochs = 0

    print(f'Training over {args.num_epochs} epochs...')

    PATH_train_loss_pkl = f"{outpath}/model_evaluations/losses_train_per_epoch"
    PATH_valid_loss_pkl = f"{outpath}/model_evaluations/losses_valid_per_epoch"
    PATH_train_acc_pkl = f"{outpath}/model_evaluations/accs_train_per_epoch"
    PATH_valid_acc_pkl = f"{outpath}/model_evaluations/accs_valid_per_epoch"
    PATH_dt_pkl = f"{outpath}/model_evaluations/dts_per_epoch"
    if not osp.isdir(PATH_train_loss_pkl):
        os.makedirs(PATH_train_loss_pkl)
    if not osp.isdir(PATH_valid_loss_pkl):
        os.makedirs(PATH_valid_loss_pkl)
    if not osp.isdir(PATH_train_acc_pkl):
        os.makedirs(PATH_train_acc_pkl)
    if not osp.isdir(PATH_valid_acc_pkl):
        os.makedirs(PATH_valid_acc_pkl)
    if not osp.isdir(PATH_dt_pkl):
        os.makedirs(PATH_dt_pkl)

    for ep in range(args.num_epochs):
        if args.load_to_train:
            epoch = ep + args.load_epoch + 1
        else:
            epoch = ep

        t0 = time.time()

        if stale_epochs > args.patience:
            print(f"Break training loop because the number of stale epochs {stale_epochs} the set patience {args.patience}.")
            break

        model.train()
        train_loss, train_acc = train(args, model, train_loader, epoch, outpath, is_train=True, optimizer=optimizer, lr=args.lr, device=device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test generalization
        model.eval()
        valid_loss, valid_acc = test(args, model, valid_loader, epoch, outpath, device=device)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = args.num_epochs - (epoch+1)
        time_per_epoch = (t1 - t0_initial)/(epoch+1)

        eta = epochs_remaining * time_per_epoch / 60

        torch.save(model.state_dict(), f"{outpath}/epoch_{epoch+1}_weights.pth")

        dt = t1-t0
        dts.append(dt)

        with open(f'{PATH_train_loss_pkl}/train_loss_epoch_{epoch}.pkl', 'wb') as f:
            pickle.dump(train_loss, f)
        with open(f'{PATH_valid_loss_pkl}/valid_loss_epoch_{epoch}.pkl', 'wb') as f:
            pickle.dump(train_loss, f)
        with open(f'{PATH_train_acc_pkl}/train_acc_epoch_{epoch}.pkl', 'wb') as f:
            pickle.dump(train_loss, f)
        with open(f'{PATH_valid_acc_pkl}/valid_acc_epoch_{epoch}.pkl', 'wb') as f:
            pickle.dump(valid_loss, f)
        with open(f'{PATH_dt_pkl}/dt_epoch_{epoch}.pkl', 'wb') as f:
            pickle.dump(valid_loss, f)

        print(f"epoch={epoch+1}/{args.num_epochs}, dt={dt}, train_loss={train_loss}, valid_loss={valid_loss}, train_acc={train_acc}, valid_acc={valid_acc}, stale_epoch(s)={stale_epochs}, eta={eta}m")

    # Recording losses and accuracies
    with open(f'{outpath}/model_evaluations/losses_train.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open(f'{outpath}/model_evaluations/accs_train.pkl', 'wb') as f:
        pickle.dump(train_accs, f)
    with open(f'{outpath}/model_evaluations/losses_valid.pkl', 'wb') as f:
        pickle.dump(valid_losses, f)
    with open(f'{outpath}/model_evaluations/accs_valid.pkl', 'wb') as f:
        pickle.dump(train_accs, f)
    with open(f'{outpath}/model_evaluations/dts.pkl', 'wb') as f:
        pickle.dump(train_accs, f)

    ### Plotting
    # Losses
    fig, ax = plt.subplots()
    ax.plot([i+1 for i in range(len(train_losses))], train_losses, label='training losses')
    ax.plot([i+1 for i in range(len(valid_losses))], valid_losses, label='validation losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation losses')
    ax.legend(loc='best')
    plt.savefig(f'{outpath}/model_evaluations/losses.{args.fig_format}')
    plt.savefig(f'{outpath}/model_evaluations/losses.png', dpi=900)
    plt.close(fig)

    # Accuracies
    fig, ax = plt.subplots()
    ax.plot([i+1 for i in range(len(train_accs))], train_accs, label='training accuracies')
    ax.plot([i+1 for i in range(len(valid_accs))], valid_accs, label='validation accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')
    plt.savefig(f'{outpath}/model_evaluations/accuracies.{args.fig_format}')
    plt.savefig(f'{outpath}/model_evaluations/accuracies.png', dpi=900)
    plt.close(fig)
