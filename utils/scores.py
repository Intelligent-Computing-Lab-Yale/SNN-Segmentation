"""

Utility functions for computing accuracy metrics.

@author: Joshua Chough

"""

import numpy as np

np.seterr(divide='ignore', invalid='ignore')

def _fast_hist(label_true, label_pred, n_class, f=None):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.asarray(np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2)).reshape(n_class, n_class)

    return hist

# Compute accuracies and mean IoU
def scores(label_trues, label_preds, n_class, batch_size, count_batch, f=None):
    count = np.zeros((batch_size, 2, n_class))
    hist = np.zeros((n_class, n_class))
    for i, (lt, lp) in enumerate(zip(label_trues, label_preds)):
        for j, (lt_ex, lp_ex) in enumerate(zip(lt, lp)):
            ex_hist = _fast_hist(lt_ex.flatten(), lp_ex.flatten(), n_class, f)
            hist += ex_hist
            if i == count_batch:
                count[j][0] = ex_hist.sum(axis=0)
                count[j][1] = ex_hist.sum(axis=1)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc': acc,
            'Mean Acc': acc_cls,
            'FreqW Acc': fwavacc,
            'Mean IoU': mean_iu,}, cls_iu, count
