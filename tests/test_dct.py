import numpy as np
import jax.numpy as jnp
from scipy.fftpack import dctn

from esn.dct import dct2
from esn.toydata import gauss2d_sequence

def sp_dct2(Fxx, nk1, nk2):
    return dctn(Fxx, norm="ortho")[:nk1,:nk2]

def test_random_dct():
    x = np.random.uniform(size=(10,10))
    y = np.array(dct2(jnp.array(x), 3, 3))
    z = sp_dct2(x, 3, 3)
    assert jnp.allclose(y,z)

def test_gauss_dct():
    x = gauss2d_sequence(size=[30,30])[0]
    y = np.array(dct2(jnp.array(x), 15, 15))
    z = sp_dct2(x , 15, 15)
    assert jnp.allclose(y,z)

if __name__ == "__main__":
    test_gauss_dct()
