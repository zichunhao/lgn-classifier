import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix, epoch, outpath, is_train, format='pdf'):
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, ax = ax) # annot=True to annotate cells
    ax.set_title(f'Confusion matrix at epoch {epoch+1}')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.invert_yaxis()
    ax.xaxis.set_ticklabels(['g', 'q', 't', 'w', 'z'])
    ax.yaxis.set_ticklabels(['g', 'q', 't', 'w', 'z'])
    if is_train:
        plt.savefig(f'{outpath}/confusion_matrix_train_epoch_{epoch+1}.{format}')
    else:
        plt.savefig(f'{outpath}/confusion_matrix_valid_epoch_{epoch+1}.{format}')
    plt.close(fig)
