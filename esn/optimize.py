import jax
import jax.numpy as jnp

#@jax.jit
def lstsq_stable(H, labels):
    if labels.ndim != 2:
        raise ValueError("Labels must have shape (time, features)")

    U, s, Vh = jax.scipy.linalg.svd(H.T)
    scale = s[0]
    n = len(s[jnp.abs(s / scale) > 1e-5])  # Ensure condition number less than 100.000
    
    L = labels.T

    v = Vh[:n, :].T
    uh = U[:, :n].T

    wout = jnp.dot(jnp.dot(L, v) / s[:n], uh)
    return wout
