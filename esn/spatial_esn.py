import numpy as np
import jax
import jax.numpy as jnp

from esn.input_map import make_input_map


def spatial_esncell(input_map_specs, hidden_size,
                   spectral_radius=1.5, density=0.1):
    """
    Create an ESN with input, and hidden weights represented as a tuple:
        esn = (Wih, Whh, bh)
    The hidden matrix (the reservoir) is a sparse matrix in turn represented
    as a tuple of values, row/column indices, and its dense shape:
        Whh = (((values, rows, cols), shape)

    Arguments:
        input_size: ESN input size
        hidden_size: ESN hidden size
        spectral_radius: spectral radius of Whh
        density: density of Whh
    Returns:
        (Wih, Whh, bh)
    """
    map_ih = make_input_map(init_input_map_specs)
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


def apply_spatial_esn(params, xs, h0):
    """
    Apply and ESN defined by params (as in created from `sparse_esncell`) to
    each input in xs with the initial state h0. Each new input uses the updated
    state from the previous step.

    Arguments:
        params: An ESN tuple (Wih, Whh, bh)
        xs: Array of inputs. Time in first size.
        h0: Initial hidden state
    Returns:
        (h,hs) where
        h: Final hidden state
        hs: All hidden states
    """
    def _step(params, h, x):
        (mapih, (Whh, shape), bh) = params
        h = jnp.tanh(sp_dot(Whh, h, shape[0]) + mapih(x) + bh)
        return (h, h)

    f = partial(_step, params)
    (h, hs) = lax.scan(f, h0, xs)
    return (h, hs)
