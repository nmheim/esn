"""
Sparse dot/matmul functions in JAX. Taken from
[here](https://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/)

The static_argnums argument in the decorators tells JAX that the 3rd argument
is static, so the functions will be compiled for every value of `shape`.
"""
import jax

@jax.partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    assert B.ndim == 2
    (values, rows, cols) = A
    in_ = B.take(cols, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, rows, shape)
    return res

@jax.partial(jax.jit, static_argnums=(2))
def sp_dot(A, x, shape):
    """
    Arguments:
        A: (N,M) sparse matrix represented as tuple (values,row,cols)
        x: (M,)  dense vector
        shape: value of N
    Returns:
        (N,) dense vector
    """
    assert x.ndim == 1
    values, rows, cols = A
    in_ = x.take(cols, axis=0)
    prod = in_*values
    res = jax.ops.segment_sum(prod, rows, shape)
    return res
