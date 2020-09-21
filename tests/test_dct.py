import numpy as np
import jax.numpy as jnp
from esn.dct import dct2
from scipy.fftpack import dctn

def sp_dct2(Fxx, nk1, nk2):
    return dctn(Fxx, norm="ortho")[:nk1,:nk2]

def test_dct():
    x = np.random.uniform(size=(10,10))
    y = np.array(dct2(jnp.array(x), 3, 3))
    z = sp_dct2(x, 3, 3)
    assert jnp.allclose(y,z)
