import numpy as np
import scipy.stats as stats
from scipy import sparse
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from esn.jaxsparse import sp_dot


def sparse_esn_reservoir(dim, spectral_radius, density, symmetric):
    """Creates a CSR representation of a sparse ESN reservoir.
    Params:
        dim: int, dimension of the square reservoir matrix
        spectral_radius: float, largest eigenvalue of the reservoir matrix
        density: float, 0.1 corresponds to approx every tenth element
            being non-zero
        symmetric: specifies if matrix.T == matrix
    Returns:
        matrix: a square scipy.sparse.csr_matrix
    """
    rvs = stats.uniform(loc=-1., scale=2.).rvs
    matrix = sparse.random(dim, dim, density=density, data_rvs=rvs)
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


def sparse_esncell(input_dim, hidden_dim,
                   spectral_radius=1.5, density=0.1):
    """
    Create an ESN with input, and hidden weights represented as a tuple:
        esn = (Wih, Whh, bh)
    The hidden matrix (the reservoir) is a sparse matrix in turn represented
    as a tuple of values, row/column indices, and its dense shape:
        Whh = (((values, rows, cols), shape)

    Arguments:
        input_dim: ESN input dimension
        hidden_dim: ESN hidden dimension
        spectral_radius: spectral radius of Whh
        density: density of Whh
    Returns:
        (Wih, Whh, bh)
    """
    Wih = np.random.uniform(-1, 1, (hidden_dim,input_dim))
    Whh = sparse_esn_reservoir(hidden_dim, spectral_radius, density, False)
    Whh = Whh.tocoo()
    bh  = np.random.uniform(-1, 1, (hidden_dim,))
    model = (jax.device_put(Wih),
             ((jax.device_put(Whh.data),
              jax.device_put(Whh.row),
              jax.device_put(Whh.col)),
             Whh.shape),
             jax.device_put(bh))
    return model

def apply_esn(params, xs, h0):
    def _step(params, x, h):
        (Wih, Whh, bh) = params
        h = jnp.tanh(Whh.dot(h) + Wih.dot(x) + bh)
        return (h, h)

    f = partial(_step, params)
    return lax.scan(f, xs, h)

def apply_sparse_esn(params, xs, h0):
    def _step(params, h, x):
        (Wih, (Whh, shape), bh) = params
        h = jnp.tanh(sp_dot(Whh, h, shape[0]) + Wih.dot(x) + bh)
        return (h, h)

    f = partial(_step, params)
    (h, hs) = lax.scan(f, h0, xs)
    return (h, hs)

def generate_state_matrix(esn, inputs, Ntrans):
    (Whh,Wih,bh) = esn
    (hidden_dim, Ntrain) = (Whh.shape[0], inputs.shape[0])
           
    h0 = jnp.zeros(hidden_dim)

    (_,H) = apply_sparse_esn(esn, inputs, h0)
    H = jnp.vstack(H)

    H0 = H[Ntrans:]
    I0 = inputs[Ntrans:]
    ones = jnp.ones((Ntrain-Ntrans,1))
    return jnp.concatenate([ones,I0,H0],axis=1)
