import numpy as np
import scipy.stats as stats
from scipy import sparse
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from esn.input_map import InputMap
from esn.jaxsparse import sp_dot
from esn.optimize import lstsq_stable
from esn.utils import _fromfile


class SparseESN:
    """
    Create an ESN with input, hidden, and output weights.
    The hidden matrix (the reservoir) is a sparse matrix represented
    as a tuple of values, and row/column indices:

        Whh = (values, rows, cols)

    The dense shape of the reservoir is (hidden_size, hidden_size).

    Arguments:
        input_map_specs: List of dicts that can be passed to
          `esn.input_map.InputMap`
        hidden_size: ESN hidden size
        spectral_radius: spectral radius of Whh
        density: density of Whh
    """

    def __init__(self, input_map_specs, hidden_size,
                 spectral_radius=1.5, density=0.1):
        self.hidden_size = hidden_size
        self.spectral_radius = spectral_radius

        self.map_ih = InputMap(input_map_specs)

        Whh = sparse_esn_reservoir(hidden_size, spectral_radius, density, False)
        Whh = Whh.tocoo()
        self.Whh = (Whh.data, Whh.row, Whh.col)
        self.bh = np.random.uniform(-1, 1, (hidden_size,))
        self.device_put()

    @classmethod
    def fromfile(cls, filename):
        return _fromfile(filename)
        
    def device_put(self):
        self.map_ih.device_put()
        (data, row, col) = self.Whh
        self.Whh = (jax.device_put(data),
                    jax.device_put(row),
                    jax.device_put(col))
        self.bh = jax.device_put(self.bh)
        if hasattr(self, "Who"):
            self.Who = jax.device_put(self.Who)

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, xs, h0):
        """
        Apply the ESN defined to each input in 'xs' with the initial state
        'h0'. Each new input uses the updated state from the previous step.

        Arguments:
            xs: Array of inputs. Time in first dimension.
            h0: Initial hidden state
        Returns:
            (h,hs) where
            h: Final hidden state
            hs: All hidden states
        """
        map_ih, Whh, bh = self.map_ih, self.Whh, self.bh
        def _step(h, x):
            h = jnp.tanh(sp_dot(Whh, h, self.hidden_size) + map_ih(x) + bh)
            return (h, h)
        (h,hs) = lax.scan(_step, h0, xs)
        return (h,hs)

    def train(self, states, labels):
        """Compute the output matrix via least squares."""
        self.Who = lstsq_stable(states, labels)
        return self.Who

    @partial(jax.jit, static_argnums=(0,3))
    def predict(self, y0, h0, Npred):
        """
        Given a trained ESN, an initial state 'h0', and input 'y0' predict in
        free-running mode for 'Npred' steps into the future, with output feeding
        back y_n as next input:
        
          h_{n+1} = \tanh(Whh h_n + Wih y_n + bh)
          y_{n+1} = Who h_{n+1}
        """
        if y0.ndim == 1:
            aug_len = y0.shape[0] + 1
        elif y0.ndim == 2:
            aug_len = y0.shape[0] * y0.shape[1] + 1
        else:
            raise ValueError("'y0' must either be a vector or a matrix.")

        def _step(input, xs):
            map_ih, Whh, bh, Who = self.map_ih, self.Whh, self.bh, self.Who

            (y,h_augmented) = input
            h = h_augmented[aug_len:]
            h = jnp.tanh(sp_dot(Whh, h, self.hidden_size) + map_ih(y) + bh)
            h = jnp.hstack([[1.], y.reshape(-1), h])
            y = Who.dot(h).reshape(y.shape)
            return ((y,h), (y,h))

        xs = jnp.arange(Npred)  # necessary for lax.scan
        ((y,h), (ys,hs)) = lax.scan(_step, (y0,h0), xs)
        return ((y,h), (ys,hs))

    def generate_state_matrix(self, inputs, Ntrans):
        Ntrain = inputs.shape[0]
               
        h0 = jnp.zeros(self.hidden_size)

        (_,H) = self.apply(inputs, h0)
        H = jnp.vstack(H)

        H0 = H[Ntrans:]
        I0 = inputs[Ntrans:].reshape(Ntrain-Ntrans,-1)
        ones = jnp.ones((Ntrain-Ntrans,1))
        return jnp.concatenate([ones,I0,H0], axis=1)


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


    map_ih = InputMap(input_map_specs)
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
