import numpy as np
from esn.utils import normalize


def gauss2d_sequence(centers=None, sigma=0.5, size=[20, 20], borders=[[-2, 2], [-2, 2]]):
    """Creates a moving gaussian blob on grid with `size`"""
    if centers is None:
        t = np.arange(0, 500 * np.pi, 0.1)
        x = np.sin(t)
        y = np.cos(0.25 * t)
        centers = np.array([y, x]).T

    yc, xc = centers[:, 0], centers[:, 1]
    yy = np.linspace(borders[0][0], borders[0][1], size[0])
    xx = np.linspace(borders[1][0], borders[1][1], size[1])

    xx = xx[None, :, None] - xc[:, None, None]
    yy = yy[None, None, :] - yc[:, None, None]

    gauss = (xx**2 + yy**2) / (2 * sigma**2)
    return np.exp(-gauss)


def square_sequence(toplefts=None, square_size=(3,3), size=(20,20), borders=[[-2, 2], [-2, 2]]):
    """Creates a moving gaussian blob on grid with `size`"""
    if toplefts is None:
        t = np.arange(0, 500 * np.pi, 0.1)
        xa = (size[1]-square_size[1]) / 2
        x  = xa * np.sin(t) + xa
        ya = (size[0]-square_size[0]) / 2
        y  = ya * np.cos(0.25 * t) + ya
        toplefts = np.rint([y, x]).T.astype(int)

    square = np.ones(square_size)
    seq    = np.zeros((toplefts.shape[0],) + size)
    sx, sy = square_size

    for (i,(cy,cx)) in enumerate(toplefts):
        seq[i, cy:cy+sy, cx:cx+sx] += square
    return seq


def mackey2d_sequence(sigma=0.5, size=[20,20], borders=[[-2, 2], [-2, 2]]):
    t = np.arange(0, 500 * np.pi, 0.1)
    x = normalize(mackey_sequence(N=t.shape[0])) * 2 - 1
    y = np.cos(t)
    centers = np.array([y,x]).T
    return gauss2d_sequence(centers, sigma=sigma, size=size, borders=borders)


def mackey_sequence(b=None, N=3000):
    """Create the Mackey-Glass series"""
    c = 0.2
    tau = 17
    n = 10

    yinit = np.array([0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076,
        1.1352, 1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756,
        1.0739, 1.0759])

    if b is None:
        b = np.zeros(N) + 0.1

    y = np.zeros(N)
    y[:yinit.shape[0]] = yinit

    for i in range(tau, N - 1):
        yi = y[i] - b[i] * y[i] + c * y[i - tau] / (1 + y[i - tau]**n)
        y[i + 1] = yi
    return y
