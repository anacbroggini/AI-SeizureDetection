import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import auc, roc_curve

def plot_confusion_matrix(y_true, y_pred, cmap='Blues'):
    """plot a confusion matrix with total and relative numbers.

    Args:
            y_true: _description_
            y_pred: _description_
    """
    
    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate relative numbers
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm.sum(axis=1)[:, np.newaxis]

    annot = np.empty_like(cm).astype(str)
    # get the dimensions
    nrows, ncols = cm.shape
    # cycle over cells and create annotations for each cell
    for i in range(nrows):
            for j in range(ncols):
                    # get the count for the cell
                    c = cm[i, j]
                    # get the percentage for the cell
                    p = cm_perc[i, j]
                    if True:
                            s = cm_sum[i]
                            # convert the proportion, count, and row sum to a string with pretty formatting
                            annot[i, j] = '%d/%d\n%.1f%%' % (c, s, p)
                    elif c == 0:
                            annot[i, j] = ''
                    else:
                            annot[i, j] = '%d\n%.1f%%' % (c, p)
    
    # convert the array to a dataframe. To plot by proportion instead of number, use cm_perc in the DataFrame instead of cm
    labels = ['No Seizure', 'Seizure']
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    
    # plot the data using the Pandas dataframe. To change the color map, add cmap=..., e.g. cmap = 'rocket_r'
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=True,
                        xticklabels=labels, yticklabels=labels)
    #plt.savefig(filename)
    plt.show()


def plot_history_metrics(history):
    for key in history.history.keys():
        plt.plot(history.history[key], label=key)

    plt.xlabel('Epoch')
    plt.ylabel('Metric value')
    plt.legend()
    plt.grid(True)


def plot_roc(y_true, y_pred):
    
    fpr, tpr, thresholds_rf = roc_curve(y_true, y_pred)
    auc_model = auc(fpr, tpr)     

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr, tpr, label='RNN (area = {:.3f})'.format(auc_model))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()