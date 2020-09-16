import time
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.linalg import svd

A = jnp.array(np.random.uniform(low=0, high=1, size=(1500,3000)))

t1 = time.time()
svd(A)
t2 = time.time()
print(f"svd: {t2-t1}")

t1 = time.time()
svd(A)
t2 = time.time()
print(f"svd: {t2-t1}")

t1 = time.time()
svd(A)
t2 = time.time()
print(f"svd: {t2-t1}")

print(svd(A))
