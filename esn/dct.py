import numpy as np
from scipy.fftpack import dct, idct, dctn, idctn
from scipy.linalg import lstsq

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from   jax.numpy import pi, exp

na = np.newaxis

# TODO: add axis
def dct_from_fft(a):
    n = a.shape[0]
    assert((n&1)==0)

    # abcdef -> ace fdb -- one step down in recursive FFT-definition to get from even level 2n-fft to full n-fft
    c  = jnp.concatenate([a[::2],a[::-2]])
    ks           = jnp.arange(n)
    omega_shift  = 2*jnp.exp(-1j*pi/(2*n)*ks)       # Can be precalculated 
    A  = jnp.fft.fft(c,axis=0) * omega_shift[:,na]

    return jnp.array(A.real)

def dctn_from_fft(a,axes=[0]):
    c = jnp.array(a)
    for axis in axes:
        c_reordered = dct_from_fft(c.swapaxes(axis,0))
        c = c_reordered.swapaxes(axis,0)
    return c
        

def sct_basis(nx, nk):
    """Basis for SCT (Slow Cosine Transform) which is a least-squares
    approximation to restricted DCT-III / Inverse DCT-II
    """
    xs = np.arange(nx)
    ks = np.arange(nk)
    basis = 2 * jnp.cos(np.pi * (xs[:, None] + 0.5) * ks[None, :] / nx)
    return basis


def sct(fx, basis):
    """SCT (Slow Cosine Transform) which is a least-squares approximation to
    restricted DCT-III / Inverse DCT-II
    """
    fk, _, _, _ = jnp.linalg.lstsq(basis, fx)
    return fk


def isct(fk, basis):
    """Inverse SCT"""
    fx = jnp.dot(basis, fk)
    return fx


def sct2(Fxx, basis1, basis2):
    """SCT of a two-dimensional array"""
    Fkx = sct(Fxx.T, basis2)
    Fkk = sct(Fkx.T, basis1)
    return Fkk


def isct2(Fkk, basis1, basis2):
    """Inverse SCT of a two-dimensional array"""
    Fkx = isct(Fkk.T, basis2)
    Fxx = isct(Fkx.T, basis1)
    return Fxx


def dct_coefficients(N):
    alpha0 = jnp.ones((1, N))
    alphaj = jnp.ones((N-1,N)) + 1
    a = jnp.sqrt(jnp.vstack([alpha0,alphaj]) / N)

    #a = jnp.sqrt(alpha0 / N)
    k,j = jnp.meshgrid(jnp.arange(N),jnp.arange(N))
    C = a * jnp.cos(jnp.pi * (2*k+1)*j / (2*N))
    return C

def dct2(Fxx, nk1, nk2):
    """Two dimensional discrete cosine transform
    Reference: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga85aad4d668c01fbd64825f589e3696d4"""
    #Fkk = dctn_from_fft(Fxx,axes=[0,1])[:nk1, :nk2]
    C = dct_coefficients(Fxx.shape[0])
    Fkk = C.dot(Fxx).dot(C.T)[:nk1,:nk2]
    return Fkk


def idct2(Fkk, nx1, nx2):
    """Two dimensional inverse discrete cosine transform"""
    Fxx = idctn(Fkk, norm='ortho', shape=(nx1, nx2))
    return Fxx


def idct2_sequence(Ftkk, xsize):
    """Inverse Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftkk : ndarray with shape (time, nk1, nk2)
    size : (ny, nx) determines the resolution of the image

    Returns
    -------
    Ftxx: ndarray with shape (time, ny, nx)
    """
    Ftxx = idctn(Ftkk, norm='ortho', shape=xsize, axes=[1, 2])
    return Ftxx


def dct2_sequence(Ftxx, ksize):
    """Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftxx : ndarray with shape (time, ydim, xdim)
    size : (nk1, nk2) determines how many DCT coefficents are kept

    Returns
    -------
    Ftkk: ndarray with shape (time, nk1, nk2)
    """
    Ftkk = dctn(Ftxx, norm='ortho', axes=[1, 2])[:, :ksize[0], :ksize[1]]
    return Ftkk
