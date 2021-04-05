import torch
import torch.nn as nn
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
import time

from plot_results import plot_confusion_matrix

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
    predictions = []
    targets = []
    correct_preds = 0
    accuracy = 0

    for i, batch in enumerate(loader):
        t0 = time.time()

        X = batch
        Y = batch['jet_types'].to(device)

        # Forward propagation
        preds = model(X)

        # Backward propagation
        loss = nn.MSELoss()
        batch_loss = loss(preds, Y.double())

        if is_train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        t1 = time.time()

        # Correction prediction
        correct_preds += (preds.argmax(axis=1) == Y.argmax(axis=1)).sum().item()
        accuracy = correct_preds / (args.batch_size*len(loader))

        if is_train:
            print(f"batch {i+1}/{len(loader)}, train_loss = {batch_loss.item()}, train_accuracy = {accuracy}, time_duration = {t1-t0}", end='\r', flush=True)
        else:
            print(f"batch {i+1}/{len(loader)}, valid_loss = {batch_loss.item()}, valid_accuracy = {accuracy}, time_duration = {t1-t0}", end='\r', flush=True)

        losses_per_epoch.append(batch_loss.item())

        # For confusion matrix
        predictions.append(preds.detach().cpu().numpy())
        targets.append(batch['jet_types'].detach().cpu().numpy())

    avg_loss_per_epoch = sum(losses_per_epoch)/len(losses_per_epoch)

    predictions = np.concatenate(predictions).argmax(axis=1)
    targets = np.concatenate(targets).argmax(axis=1)

    confusion_matrix = sklearn.metrics.confusion_matrix(targets, predictions, normalize='true')
    confusion_matrix[[0, 1],:] = confusion_matrix[[1, 0],:]  # swap rows for better visualization of confusion matrix

    if is_train:
        with open(f"{outpath}/valid_accuracy_epoch_{epoch+1}.pkl", 'wb') as f:
            pickle.dump(accuracy, f)
        plot_confusion_matrix(confusion_matrix, epoch, outpath, is_train=True, format=args.fig_format)
    else:
        plot_confusion_matrix(confusion_matrix, epoch, outpath, is_train=False, format=args.fig_format)

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

    best_valid_loss = math.inf
    stale_epochs = 0

    print(f'Training over {args.num_epochs} epochs...')

    for epoch in range(args.num_epochs):
        t0 = time.time()

        if stale_epochs > args.patience:
            print(f"Break training loop because the number of stale epochs {stale_epochs} the set patience {args.patience}.")
            break

        model.train()
        train_loss, train_acc = train(args, model, train_loader, epoch, outpath, is_train=True, optimizer=optimizer, lr=args.lr_init, device=device)
        train_losses.append(train_loss)

        # Test generalization
        model.eval()
        valid_loss, valid_acc = test(args, model, valid_loader, epoch, outpath, device=device)
        valid_losses.append(valid_loss)

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

        print(f"epoch={epoch+1}/{args.num_epochs}, dt={t1-t0}, train_loss={train_loss}, valid_loss={valid_loss}, train_acc={train_acc}, valid_acc={valid_acc}, stale_epoch(s)={stale_epochs}, eta={eta}m")


    fig, ax = plt.subplots()
    ax.plot([i+1 for i in range(len(train_losses))], train_losses, label='train losses')
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.savefig(f'{outpath}/train_losses.{args.fig_format}')
    plt.close(fig)

    with open(outpath + '/train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)

    fig, ax = plt.subplots()
    ax.plot([i+1 for i in range(len(valid_losses))], valid_losses, label='train losses')
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.savefig(f'{outpath}/valid_losses.{args.fig_format}')
    plt.close(fig)

    with open(f'{outpath}/valid_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
