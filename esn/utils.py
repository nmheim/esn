import joblib
import numpy as np

def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length + 1
    train_seq = sequence[:train_end]
    inputs = train_seq[:-1]
    labels = train_seq[1:]
    pred_labels = sequence[train_end:train_end + pred_length]
    return inputs, labels, pred_labels


def normalize(data, vmin=None, vmax=None):
    """Normalizes data to values from 0 to 1.
    If vmin/vmax are given they are assumed to be the maximal
    values of data"""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    if vmin==vmax:
        return np.zeros_like(data)
    else:
        return (data - vmin) / (vmax-vmin)


def _fromfile(filename):
    with open(filename, "rb") as fi:
        m = joblib.load(fi)
        m.device_put()
    return m
