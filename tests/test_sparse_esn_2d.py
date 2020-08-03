import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from esn.sparse_esn import (sparse_esncell,
                            sparse_generate_state_matrix,
                            predict_sparse_esn)
from esn.input_map import map_output_size
from esn.utils import split_train_label_pred
from esn.toydata import gauss2d_sequence
from esn.optimize import lstsq_stable


def sparse_esn_2d_train_pred(data, specs,
                             Ntrans=500,        # Number of transient initial steps before training
                             Ntrain=2500,       # Number of steps to train on
                             Npred=500,         # Number of steps for free-running prediction
                             mse_threshold=1e-5,
                             plot_prediction=False):
    N = Ntrain + Npred + 1
    assert data.ndim == 3
    assert data.shape[0] >= N

    inputs, labels, pred_labels = split_train_label_pred(data,Ntrain,Npred)

    hidden_size = map_output_size(specs, inputs.shape[1:])
    esn = sparse_esncell(specs, hidden_size, spectral_radius=1.5, density=0.05)
    
    inputs = inputs.reshape(inputs.shape[0], -1)
    labels = labels.reshape(inputs.shape[0], -1)
    
    H = sparse_generate_state_matrix(esn, inputs, Ntrans)
    Who = lstsq_stable(H, labels[Ntrans:])
    model = esn + (Who,)
    
    y0 = labels[-1]
    h0 = H[-1]
    (y,h), (ys,hs) = predict_sparse_esn(model, y0, h0, Npred)
    
    ys = ys.reshape(*pred_labels.shape)
    
    if plot_prediction:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(pred_labels.sum(axis=0))
        ax[0].set_title("Truth")
        ax[1].imshow(ys.sum(axis=0))
        ax[1].set_title("Pred.")
        plt.show()

    mse = jnp.mean(jnp.abs(ys[-1] - pred_labels[-1]))
    assert mse < mse_threshold


def test_sparse_esn_lissajous():
    input_shape = (20,20)
    input_size  = input_shape[0] * input_shape[1]

    data = gauss2d_sequence(size=input_shape)
    specs = [{"type":"random_weights",
              "input_size":input_size,
              "hidden_size":3500,
              "factor": 1.0}]

    sparse_esn_2d_train_pred(data, specs,
                             Npred=200,
                             plot_prediction=False,
                             mse_threshold=1e-4)

if __name__ == "__main__":
    test_sparse_esn_lissajous()
