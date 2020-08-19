import pytest
import numpy as np
from scipy.signal import convolve2d
from esn.imed import imed, imed_matrix


def gauss_kernel(kernel_shape, sigma=None):
    if sigma is None:
        sigma = min(kernel_shape) / 5.
    ysize, xsize = kernel_shape
    yy = np.linspace(-int(ysize / 2.), int(ysize / 2.), ysize)
    xx = np.linspace(-int(xsize / 2.), int(xsize / 2.), xsize)

    P = xx[:, None]**2 + yy[None, :]**2
    gaussian = np.exp(-P / (2 * sigma**2))
    gaussian = 1. / (2 * np.pi * sigma**2) * gaussian
    return gaussian


def test_imed():
    np.random.seed(0)

    img1 = np.zeros([15,15])
    img1[7:12,7:12] = 1.
    img2 = np.zeros([15,15])
    img2[8:13,8:13] = 1.
    diff = img1 - img2
    sigma = 2.

    kernel = gauss_kernel([15,15], sigma=sigma)
    conv_diff = convolve2d(diff, kernel, mode="same")

    G = imed_matrix(diff.shape, sigma=sigma)
    G_diff = G.dot(diff.reshape(-1)).reshape(diff.shape)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(kernel)
    # ax[1].imshow(G)

    # fig2, ax2 = plt.subplots(2,2)
    # ax2 = ax2.flatten()
    # ax2[0].imshow(img1)
    # ax2[0].set_title("img1")
    # ax2[1].imshow(img2)
    # ax2[1].set_title("img2")
    # ax2[2].imshow(conv_diff)
    # ax2[2].set_title("conv_diff")
    # ax2[3].imshow(G_diff)
    # ax2[3].set_title("G_diff")
    # plt.show()

    assert np.allclose(conv_diff, G_diff, atol=1e-4)

    _imed = imed(img1[None,:,:], img2[None,:,:], G=G, sigma=sigma)
    conv_diff = (diff * conv_diff).sum()

    assert np.allclose(_imed, conv_diff)

if __name__ == "__main__":
    test_imed()
