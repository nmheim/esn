import jax
import jax.numpy as jnp
from esn.imed import imed_matrix


#@jax.jit
# NOTE: this does not work because dynamic slicing is not suppored in jitted
#       functions. 'n' is the dynamic part here...
def lstsq_stable(H, labels, thresh=1e-4):
    if labels.ndim != 2:
        raise ValueError("Labels must have shape (time, features)")

    U, s, Vh = jax.scipy.linalg.svd(H.T, full_matrices=False)
    scale = s[0]
    n = jnp.sum(jnp.abs(s/scale) > thresh)  # Ensure condition number less than 1/thresh
    
    L = labels.T
    v = Vh[:n, :].T
    uh = U[:, :n].T
    
    del U

    wout = jnp.dot(jnp.dot(L, v) / s[:n], uh)
    return wout

# NOTE: maybe this will help with dynamic slicing in the future?
# from jax.lax import dynamic_slice
# from functools import partial
# @partial(jax.jit, static_argnums=(4,))
# def _compute_wout(L, Vh, s, U, n):
#     v  = dynamic_slice(Vh, (0,0), (n,Vh.shape[1])).T
#     uh = dynamic_slice(U, (0,0), (U.shape[1],n)).T
#     s_ = dynamic_slice(s, (0,), (n,))
#     wout = jnp.dot(jnp.dot(L, v) / s_, uh)
#     return wout


def imed_lstsq_stable(states, inputs, labels, sigma):
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    flat_labels = labels.reshape(inputs.shape[0], -1)

    # prep IMED
    G = imed_matrix(inputs.shape[1:], sigma=sigma)
    (w,V) = jnp.linalg.eigh(G)

    #thresh = 0
    #n = jnp.sum(jnp.abs(w) < thresh)
    #s = jnp.sqrt(w[n:])
    #V = V[n:,:]
    s = jnp.sqrt(w)

    G12 = jnp.dot(V.T, s[:,None]*V)

    # transform inputs / labels
    flat_inputs = jnp.matmul(G12, flat_inputs[:,:,None])[:,:,0]
    flat_labels = jnp.matmul(G12, flat_labels[:,:,None])[:,:,0]

    # compute Who
    Who  = lstsq_stable(states, flat_labels)
    s    = 1/s
    iG12 = jnp.dot(V.T,s[:,None]*V)
    Who  = jnp.dot(iG12, Who)
    return Who
