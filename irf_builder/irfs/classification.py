import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc

import irf_builder as irf


def get_roc_curve(events):

    y_test, y_score = [], []
    for ch, ev in events.items():
        y_score += ev["gammaness"].values.tolist()
        y_test += [ch] * len(ev)

    fpr, tpr, _ = roc_curve(np.array(y_test) == "g", y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc):

    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')

    # plot a diagonal line that represents purely random choices
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # plot a horizontal line at y = 1
    plt.plot([0, 1], [1, 1], color='gray', lw=1, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend()
    plt.tight_layout()
