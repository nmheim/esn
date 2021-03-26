import pytest
import joblib
import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from esn.input_map import InputMap
from esn.utils import split_train_label_pred
from esn.toydata import gauss2d_sequence, mackey2d_sequence
from esn.imed import imed
import esn.sparse_esn as se


def sparse_esn_2d_train_pred(tmpdir, data, specs,
                             spectral_radius=1.5,
                             density=0.01,
                             Ntrans=500,        # Number of transient initial steps before training
                             Ntrain=2500,       # Number of steps to train on
                             Npred=500,         # Number of steps for free-running prediction
                             mse_threshold=1e-5,
                             plot_prediction=False):
    np.random.seed(1)
    N = Ntrain + Npred + 1
    assert data.ndim == 3
    assert data.shape[0] >= N

    # prepare data
    inputs, labels, pred_labels = split_train_label_pred(data,Ntrain,Npred)
    img_shape = inputs.shape[1:]

    # build esn
    map_ih = InputMap(specs)
    hidden_size = map_ih.output_size(img_shape)
    print("Hidden size: ", hidden_size)
    esn = se.esncell(map_ih, hidden_size, spectral_radius=spectral_radius, density=density)

    # compute training states
    H = se.augmented_state_matrix(esn, inputs, Ntrans)

    # compute last layer without imed
    _labels = labels.reshape(inputs.shape[0], -1)
    model = se.train(esn, H, _labels[Ntrans:])
    # and with imed
    model = se.train_imed(esn, H, inputs[Ntrans:], labels[Ntrans:], sigma=2.)
    
    # predict
    y0, h0 = labels[-1], H[-1]
    (y,h), (ys,hs) = se.predict(model, y0, h0, Npred)
    # predict with warump of Ntrain frames
    _, (wys,_) = se.warmup_predict(model, labels[-Ntrans:], Npred)

    if plot_prediction:
        import matplotlib.pyplot as plt
        from AnomalyDetectionESN.visualize import animate_double_imshow
        anim = animate_double_imshow(ys, pred_labels)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(pred_labels.sum(axis=0))
        ax[0].set_title("Truth")
        ax[1].imshow(ys.sum(axis=0))
        ax[1].set_title("Pred.")
        plt.show()

    mse = jnp.mean((ys[25] - pred_labels[25])**2)
    w_mse = jnp.mean((wys[25] - pred_labels[25])**2)
    print("MSE:  ", mse)
    # print("IMED: ", imed(ys, pred_labels)[25])
    assert mse < mse_threshold
    assert w_mse < mse_threshold
    assert jnp.isclose(mse, w_mse, atol=1e-3)
"""
    with open(tmpdir / "esn.pkl", "wb") as fi:
        joblib.dump(model, fi)
    pkl_model = se.load_model(tmpdir / "esn.pkl")
    _, (pkl_ys,_) = se.predict(pkl_model, y0, h0, Npred)
    print(ys[0])
    print(pkl_ys)
    assert jnp.all(jnp.isclose(pkl_ys, ys))
"""


def test_sparse_esn_lissajous(tmpdir):
    input_shape = (30,30)
    input_size  = input_shape[0] * input_shape[1]

    from esn.utils import scale
    data = gauss2d_sequence(size=input_shape)
    data = scale(data, -1, 1)
    specs = [
        {"type": "pixels", "size": [30, 30], "factor": 3.},
        {"type": "dct", "size": [15, 15], "factor": 1.},
        {"type": "gradient", "factor": 4.},
    ]

    sparse_esn_2d_train_pred(tmpdir, data, specs,
        plot_prediction=False, mse_threshold=1e-15,
        spectral_radius=2.0, density=0.01,
        Ntrain=2000, Npred=300, Ntrans=500)


def test_sparse_esn_chaotic(tmpdir):
    input_shape = (30,30)
    input_size  = input_shape[0] * input_shape[1]

    data = mackey2d_sequence(size=input_shape)

    specs = [
        {"type":"pixels", "size":input_shape, "factor": 3.},
        {"type":"conv", "size":(3,3),   "kernel":"gauss",  "factor": 2.},
        {"type":"conv", "size":(5,5),   "kernel":"gauss",  "factor": 2.},
        {"type":"conv", "size":(7,7),   "kernel":"gauss",  "factor": 2.},
        {"type":"conv", "size":(9,9),   "kernel":"gauss",  "factor": 2.},
        {"type":"conv", "size":(3,3),   "kernel":"random", "factor": 2.},
        {"type":"conv", "size":(5,5),   "kernel":"random", "factor": 2.},
        {"type":"conv", "size":(7,7),   "kernel":"random", "factor": 2.},
        {"type":"conv", "size":(9,9),   "kernel":"random", "factor": 2.},
        {"type":"gradient", "factor": 2.},
        {"type":"dct", "size":(15,15), "factor": 0.1},
        {"type":"random_weights", "input_size":input_size, "hidden_size":3500, "factor": 1.},
    ]

    sparse_esn_2d_train_pred(tmpdir, data, specs,
        plot_prediction=False, mse_threshold=1e-2,
        spectral_radius=2.0, density=0.01,
        Ntrain=2500, Npred=300, Ntrans=500)



if __name__ == "__main__":
    test_sparse_esn_lissajous("tmp")
