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

    shape = [15,15]
    img1 = np.zeros(shape)
    img1[7:12,7:12] = 1.
    img2 = np.zeros(shape)
    img2[8:13,8:13] = 1.
    diff = img1 - img2
    sigma = 2.

    kernel = gauss_kernel(shape, sigma=sigma)
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

    print(conv_diff - G_diff)
    assert np.allclose(conv_diff, G_diff, atol=1e-4)

    _imed = imed(img1[None,:,:], img2[None,:,:], G=G, sigma=sigma)
    conv_diff = (diff * conv_diff).sum()

    print(conv_diff - _imed)
    assert np.allclose(_imed, conv_diff)

def test_imed_positive_definite():
    # TODO: fails for s = 3.
    for s in [0.5, 1., 2.]:
        G = imed_matrix((30,30), sigma=s)
        eigs = np.linalg.eigvals(G)
        neg_eigs = eigs[eigs<0]
        assert len(neg_eigs) == 0


# def test_imed_sequence():
#     import esn.toydata as td
#     imgs = td.gauss2d_sequence()[:300]
#     imgs = td.square_sequence()[:300]
#     img0 = np.tile(imgs[0], (imgs.shape[0], 1, 1))
# 
#     il = imed(imgs, img0, sigma=2.)
#     ml = ((imgs - img0)**2).mean(axis=-1).mean(axis=-1)
# 
#     import matplotlib.pyplot as plt
#     plt.plot(il/il.max(), label="imed")
#     plt.plot(ml/ml.max(), label="mse")
#     plt.legend()
#     plt.show()

if __name__ == "__main__":
    test_imed_sequence()
