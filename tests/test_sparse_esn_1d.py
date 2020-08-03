import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from esn.sparse_esn import (sparse_esncell,
                     sparse_generate_state_matrix,
                     predict_sparse_esn)
from esn.utils import split_train_label_pred
from esn.toydata import mackey_sequence
from esn.optimize import lstsq_stable


def sparse_esn_1d_train_pred(data,
                             Ntrans=500,        # Number of transient initial steps before training
                             Ntrain=2500,       # Number of steps to train on
                             Npred=500,         # Number of steps for free-running prediction
                             hidden_size=1500,  # size of reservoir
                             mse_threshold=1e-10,
                             plot_states=False,
                             plot_trained_outputs=False,
                             plot_prediction=False):
    N = Ntrain + Npred + 1
    assert data.ndim == 1
    assert data.shape[0] >= N

    input_size  = 1

    specs = [{"type":"random_weights",
              "input_size":input_size,
              "hidden_size":hidden_size,
              "factor": 1.0}]
    esn = sparse_esncell(specs, hidden_size, spectral_radius=1.5, density=0.05)

    data = data.reshape(-1, 1)
    inputs, labels, pred_labels = split_train_label_pred(data,Ntrain,Npred)

    H = sparse_generate_state_matrix(esn, inputs, Ntrans)
    assert H.shape == (Ntrain-Ntrans, hidden_size+input_size+1)
    if plot_states:
        import matplotlib.pyplot as plt
        plt.plot(H)
        plt.show()

    Who = lstsq_stable(H, labels[Ntrans:])
    Wih, Whh, bh = esn
    model = (Wih,Whh,bh,Who)
    if plot_trained_outputs:
        import matplotlib.pyplot as plt
        plt.plot(labels[Ntrans:])
        ts = Who.dot(H.T).reshape(-1)
        plt.plot(ts)
        plt.show()

    y0 = labels[-1]
    h0 = H[-1]
    (y,h), (ys,hs) = predict_sparse_esn(model, y0, h0, Npred)
    assert y.shape == (1,)
    assert ys.shape == (Npred, 1)
    assert h.shape == (hidden_size+input_size+1,)
    assert hs.shape == (Npred, hidden_size+input_size+1)

    if plot_prediction:
        import matplotlib.pyplot as plt
        plt.plot(ys, label="Truth")
        plt.plot(pred_labels.reshape(-1), label="Prediction")
        plt.title("500 step prediction vs. truth")
        plt.legend()
        plt.show()

    mse = jnp.mean((ys - pred_labels)**2)
    assert mse < mse_threshold


def test_sparse_esn_sines():
    Ntrain = 2500
    Npred  = 500
    xs   = jnp.linspace(0,30*2*jnp.pi,Ntrain+Npred+1)
    data = jnp.sin(xs)
    sparse_esn_1d_train_pred(data, Ntrain=Ntrain, Npred=Npred)


def test_sparse_esn_mackey():
    data = mackey_sequence(N=3500)
    sparse_esn_1d_train_pred(data,
                             hidden_size=2000,
                             Npred=200,
                             plot_prediction=False,
                             mse_threshold=1e-4)

if __name__ == "__main__":
    test_sparse_esn_mackey()
    #test_sparse_esn_sines()
