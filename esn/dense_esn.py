import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def dense_esn_reservoir(size, spectral_radius, density):
    pass


def dense_esncell(input_size, hidden_size, spectral_radius=1.5, density=0.1):
    pass


def dense_apply_esn(params, xs, h0):
    def _step(params, x, h):
        (Wih, Whh, bh) = params
        h = jnp.tanh(Whh.dot(h) + Wih.dot(x) + bh)
        return (h, h)

    f = partial(_step, params)
    (h, hs) = lax.scan(f, xs, h)
    return (h, hs)


def dense_predict_esn(model, y0, h0, Npred):
    (Wih,Whh,bh,Who) = model
    aug_len = y0.shape[0]+1  # augment hidden state h

    def _step(params, input, xs):
        (Wih,Whh,bh,Who) = params
        (y,h_augmented) = input
        aug, h = h_augmented[:aug_len], h_augmented[aug_len:]
        h = jnp.tanh(Whh.dot(h) + Wih.dot(y) + bh)
        h = jnp.hstack([aug, h])
        y = Who.dot(h)
        return ((y,h), (y,h))

    xs = jnp.arange(Npred)  # necessary for lax.scan
    f = partial(_step, model)
    ((y,h), (ys,hs)) = lax.scan(f, (y0,h0), xs)
    return ((y,h), (ys,hs))


def dense_generate_state_matrix(esn, inputs, Ntrans):
    (Whh,Wih,bh) = esn
    (hidden_size, Ntrain) = (Whh.shape[0], inputs.shape[0])
           
    h0 = jnp.zeros(hidden_size)

    (_,H) = dense_apply_esn(esn, inputs, h0)
    H = jnp.vstack(H)

    H0 = H[Ntrans:]
    I0 = inputs[Ntrans:]
    ones = jnp.ones((Ntrain-Ntrans,1))
    return jnp.concatenate([ones,I0,H0],axis=1)
