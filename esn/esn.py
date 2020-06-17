import jax.numpy as jnp
from jax import lax

def sparse_esn_reservoir(dim, spectral_radius, density, symmetric):
    """Creates a CSR representation of a sparse ESN reservoir.
    Params:
        dim: int, dimension of the square reservoir matrix
        spectral_radius: float, largest eigenvalue of the reservoir matrix
        density: float, 0.1 corresponds to approx every tenth element
            being non-zero
        symmetric: specifies if matrix.T == matrix
    Returns:
        matrix: a square scipy.sparse.csr_matrix
    """
    rvs = uniform(loc=-1., scale=2.).rvs
    matrix = sparse.random(dim, dim, density=density, data_rvs=rvs)
    matrix = matrix.tocsr()
    if symmetric:
        matrix = sparse.triu(matrix)
        tril = sparse.tril(matrix.transpose(), k=-1)
        matrix = matrix + tril
        # calc eigenvalues with scipy's lanczos implementation:
        eig, _ = sparse.linalg.eigsh(matrix, k=2, tol=1e-4)
    else:
        eig, _ = sparse.linalg.eigs(matrix, k=2, tol=1e-4)

    rho = np.abs(eig).max()
    matrix = matrix.multiply(1. / rho)
    matrix = matrix.multiply(spectral_radius)
    return matrix


def sparse_esncell(input_dim, hidden_dim,
                   spectral_radius=1.5, density=0.1):
    Wih = jnp.random.uniform(-1, 1, (input_dim,1))
    Whh = sparse_esn_reservoir(hidden_dim, spectral_radius, density, False)
    bh  = jnp.random.uniform(-1, 1, (hidden_dim,1))
    return (Wih, Whh, bh)

def apply(xs, h0):
    
    def _step(params, x, h):
        (Wih, Whh, bh) = params
        h = Whh.dot(h) + Wih.dot(x)
        return (h, h)

    f = partial(_step, params)
    return lax.scan(f, xs, h)

def apply_esn(params, x, h):



# class SparseJAXESNCell():
#     def __init__(self, )
# 
# 
# class ESN(nn.Module):
# 
#     def apply(self, x, h,
#               density=0.1,
#               spectral_radius=1.5)
# 
#         hidden_dim = h.shape[0]
# 
#         Wih = jnp.random.uniform(-1, 1, ())
#         Whh = sparse_esn_reservoir(hidden_dim, spectral_radius, density, False)
#         bh  = jnp.random.uniform(-1, 1, hidden_dim)
# 
#         return np.tanh(Whh.dot(h) + Wih.dot(inputs[i]) + bh)
