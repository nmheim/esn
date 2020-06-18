import numpy as np
import jax.numpy as jnp
from scipy import sparse

from esn.jaxsparse import sp_dot, sp_matmul

def test_sp_dot():
    x = np.random.uniform(-1, 1, 100)
    shape = (50,100)
    A_dense = np.random.uniform(-1, 1, shape)
    y_dense = A_dense.dot(x)
    A_sparse = sparse.coo_matrix(A_dense)
    A_sparse = (A_sparse.data, A_sparse.row, A_sparse.col)
    y_sparse = np.asarray(sp_dot(A_sparse, x, shape[0]))
    assert np.all(np.isclose(y_dense, y_sparse))

def test_matmul():
    B = np.random.uniform(-1, 1, (100,40))
    shape = (50,100)
    A_dense = np.random.uniform(-1, 1, shape)
    C_dense = A_dense.dot(B)
    A_sparse = sparse.coo_matrix(A_dense)
    A_sparse = (A_sparse.data, A_sparse.row, A_sparse.col)
    C_sparse = np.asarray(sp_matmul(A_sparse, B, shape[0]))
    assert np.all(np.isclose(C_dense, C_sparse, atol=1e-5))
