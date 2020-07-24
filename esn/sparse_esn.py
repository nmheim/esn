import numpy as np
import scipy.stats as stats
from scipy import sparse
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from esn.input_map import make_input_map
from esn.jaxsparse import sp_dot


def sparse_esn_reservoir(size, spectral_radius, density, symmetric):
    """Creates a CSR representation of a sparse ESN reservoir.
    Params:
        size: int, size of the square reservoir matrix
        spectral_radius: float, largest eigenvalue of the reservoir matrix
        density: float, 0.1 corresponds to approx every tenth element
            being non-zero
        symmetric: specifies if matrix.T == matrix
    Returns:
        matrix: a square scipy.sparse.csr_matrix
    """
    rvs = stats.uniform(loc=-1., scale=2.).rvs
    matrix = sparse.random(size, size, density=density, data_rvs=rvs)
    matrix = matrix.tocsr()
    if symmetric:
        matrix = sparse.triu(matrix)
        tril = sparse.tril(matrix.transpose(), k=-1)
        matrix = matrix + tril
        # calc eigenvalues with scipy's lanczos implementation:
        eig, _ = sparse.linalg.eigsh(matrix, k=2, tol=1e-4)
    else:
        eig, _ = sparse.linalg.eigs(matrix, k=2, tol=1e-4)

    rho = np.abs(eig).max()
    matrix = matrix.multiply(1. / rho)
    matrix = matrix.multiply(spectral_radius)
    return matrix


def sparse_esncell(input_map_specs, hidden_size,
                   spectral_radius=1.5, density=0.1):
    """
    Create an ESN with input, and hidden weights represented as a tuple:
        esn = (Wih, Whh, bh)
    The hidden matrix (the reservoir) is a sparse matrix in turn represented
    as a tuple of values, row/column indices, and its dense shape:
        Whh = (((values, rows, cols), shape)

    Arguments:
        input_map_specs: List of dicts that can be passed to
          `esn.input_map.make_input_map`
        hidden_size: ESN hidden size
        spectral_radius: spectral radius of Whh
        density: density of Whh
    Returns:
        (Wih, Whh, bh)
    """
    map_ih = make_input_map(input_map_specs)
    Whh = sparse_esn_reservoir(hidden_size, spectral_radius, density, False)
    Whh = Whh.tocoo()
    bh  = np.random.uniform(-1, 1, (hidden_size,))
    model = (map_ih,
             ((jax.device_put(Whh.data),
              jax.device_put(Whh.row),
              jax.device_put(Whh.col)),
             Whh.shape),
             jax.device_put(bh))
    return model

def apply_sparse_esn(model, xs, h0):
    """
    Apply and ESN defined by model (as in created from `sparse_esncell`) to
    each input in xs with the initial state h0. Each new input uses the updated
    state from the previous step.

    Arguments:
        model: An ESN tuple (Wih, Whh, bh)
        xs: Array of inputs. Time in first dimension.
        h0: Initial hidden state
    Returns:
        (h,hs) where
        h: Final hidden state
        hs: All hidden states
    """
    def _step(model, h, x):
        (map_ih, (Whh, shape), bh) = model
        h = jnp.tanh(sp_dot(Whh, h, shape[0]) + map_ih(x) + bh)
        return (h, h)

    f = partial(_step, model)
    (h, hs) = lax.scan(f, h0, xs)
    return (h, hs)

def predict_sparse_esn(model, y0, h0, Npred):
    """
    Given a trained model = (Wih,Whh,bh,Who), a start internal state h0, and input
    y0 predict in free-running mode for Npred steps into the future, with
    output feeding back y_n as next input:
    
      h_{n+1} = \tanh(Whh h_n + Wih y_n + bh)
      y_{n+1} = Who h_{n+1}
    """
    aug_len = y0.shape[0]+1

    def _step(params, input, xs):
        (map_ih,(Whh,shape),bh,Who) = params
        (y,h_augmented) = input
        h = h_augmented[aug_len:]
        h = jnp.tanh(sp_dot(Whh, h, shape[0]) + map_ih(y) + bh)
        h = jnp.hstack([[1.], y, h])
        y = Who.dot(h)
        return ((y,h), (y,h))

    xs = jnp.arange(Npred)  # necessary for lax.scan
    f = partial(_step, model)
    ((y,h), (ys,hs)) = lax.scan(f, (y0,h0), xs)
    return ((y,h), (ys,hs))

def get_hidden_size(esn):
    (_,(Whh,shape),_) = esn
    return shape[0]

def sparse_generate_state_matrix(esn, inputs, Ntrans):
    Ntrain = inputs.shape[0]
           
    h0 = jnp.zeros(get_hidden_size(esn))

    (_,H) = apply_sparse_esn(esn, inputs, h0)
    H = jnp.vstack(H)

    H0 = H[Ntrans:]
    I0 = inputs[Ntrans:]
    ones = jnp.ones((Ntrain-Ntrans,1))
    return jnp.concatenate([ones,I0,H0], axis=1)
