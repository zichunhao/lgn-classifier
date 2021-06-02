import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import os
import os.path as osp


def plot_confusion_matrix(args, confusion_matrix, epoch, outpath, is_train):
    PATH = f'{outpath}/model_evaluations/confusion_matrices'
    if not osp.isdir(PATH):
        os.makedirs(PATH)

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, ax=ax, vmin=0, vmax=1)
    ax.set_title(f'Confusion matrix at epoch {epoch+1}')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.invert_yaxis()  # y axis increasing upward
    ax.xaxis.set_ticklabels(args.class_labels)
    ax.yaxis.set_ticklabels(args.class_labels)
    if is_train:
        plt.savefig(f'{PATH}/confusion_matrix_train_epoch_{epoch+1}.{args.fig_format}', transparent=True)
    else:
        plt.savefig(f'{PATH}/confusion_matrix_valid_epoch_{epoch+1}.{args.fig_format}', transparent=True)

    plt.close(fig)


def plot_roc_curve(args, predictions_onehot, targets_onehot, epoch, outpath, is_train):

    PATH = f'{outpath}/model_evaluations/roc_curves'
    if not osp.isdir(PATH):
        os.makedirs(PATH)

    tpr, fpr, _, roc_auc = find_tpr_fpr_threshold_rocAUC(args, predictions_onehot, targets_onehot)

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {round(roc_auc["micro"],4)})',
             color='gold', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {round(roc_auc["macro"],4)})',
             color='crimson', linestyle=':', linewidth=4)

    # Color scheme from https://stackoverflow.com/questions/51694827/matplotlib-nice-plot-who-knows-the-scheme-used
    colors = cycle(["#7aa0c4", "#ca82e1", "#df9f53", "#64b9a1", "#745ea6"])

    for i, color in zip(range(args.num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f'ROC curve of {args.class_labels[i]} jet (area = {round(roc_auc[i], 4)})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic (ROC) to multi-class jet classification at epoch {epoch+1}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if is_train:
        plt.savefig(f'{PATH}/ROC_train_epoch_{epoch+1}.{args.fig_format}', bbox_inches='tight', transparent=True)
    else:
        plt.savefig(f'{PATH}/ROC_valid_epoch_{epoch+1}.{args.fig_format}', bbox_inches='tight', transparent=True)

    return tpr, fpr, roc_auc


def find_tpr_fpr_threshold_rocAUC(args, predictions_onehot, targets_onehot):
    fpr = {}
    tpr = {}
    threshold = {}
    roc_auc = {}

    for i in range(args.num_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(targets_onehot[:, i], predictions_onehot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], threshold["micro"] = roc_curve(targets_onehot.ravel(), predictions_onehot.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_classes)]))

    # Finally average it and compute AUC
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(args.num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= args.num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return tpr, fpr, threshold, roc_auc
