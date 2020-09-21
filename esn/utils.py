import joblib
import numpy as np

def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length + 1
    train_seq = sequence[:train_end]
    inputs = train_seq[:-1]
    labels = train_seq[1:]
    pred_labels = sequence[train_end:train_end + pred_length]
    return inputs, labels, pred_labels


def scale(x, a, b):
    """Scale array 'x' to values in (a,b)"""
    mi, ma = x.min(), x.max()
    return (b-a) * (x - mi) / (ma-mi) + a

def normalize(x):
    """Normalize array 'x' to values in (0,1)"""
    mi, ma = x.min(), x.max()
    return (x - mi) / (ma-mi)

def _fromfile(filename):
    with open(filename, "rb") as fi:
        m = joblib.load(fi)
        m.device_put()
    return m
