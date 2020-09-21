import numpy as np
import jax.numpy as jnp
from esn.utils import scale, normalize


def test_normalize():
    x = jnp.array(np.random.normal(size=(3,3)))
    y = normalize(x)
    assert np.isclose(y.min(), 0)
    assert np.isclose(y.max(), 1)

def test_scale():
    a = -np.random.uniform(0, 10)
    b = np.random.uniform(0, 10)
    x = jnp.array(np.random.normal(size=(3,3)))
    y = scale(x, a, b)
    assert np.isclose(y.min(), a)
    assert np.isclose(y.max(), b)

