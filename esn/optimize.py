import jax
import jax.numpy as jnp
from esn.imed import imed_matrix


#@jax.jit  # TODO: this does not work yet because of dynamic shapes due to 'n'
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


#@jax.jit  # TODO: this does not work yet because of dynamic shapes due to 'n'
def imed_lstsq_stable(states, inputs, labels, sigma):
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    flat_labels = labels.reshape(inputs.shape[0], -1)

    # prep IMED
    G = imed_matrix(inputs.shape[1:], sigma=sigma)
    (w,V) = jnp.linalg.eigh(G)
    s = jnp.sqrt(w)
    G12 = jnp.dot(V, s[:,None]*V.T)

    # transform inputs / labels
    flat_inputs = jnp.matmul(G12, flat_inputs[:,:,None])[:,:,0]
    flat_labels = jnp.matmul(G12, flat_labels[:,:,None])[:,:,0]

    # compute Who
    Who  = lstsq_stable(states, flat_labels)
    s    = 1/jnp.sqrt(w)
    iG12 = jnp.dot(V,s[:,None]*V.T)
    Who  = jnp.dot(iG12, Who)
    return Who
