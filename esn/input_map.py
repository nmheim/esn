from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax import image

from esn.dct import dct2
from esn.utils import _fromfile

class Operation:

    @classmethod
    def fromfile(cls, filename):
        return _fromfile(filename)

    def device_put(self):
        pass


"""
Creates a callable object, that can be called with a 2D input image.
The `InputMap` is composed of a number of `operation`s.
Each `operation` again takes an image as input and outputs a vector.
Possible operations include convolutions, random maps, resize, etc. For a full
list of operations and their specifications see `make_operation`.

Params:
    specs: list of dicts with that each specify an `operation`

Returns:
    A function that can be called with a 2D array and that outputs
    a 1D array (concatenated output of each op).
"""
class InputMap(Operation):
    def __init__(self, specs):
        self.ops = [make_operation(s) for s in specs]

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return jnp.concatenate([op(img) for op in self.ops], axis=0) 

    def device_put(self):
        for op in self.ops: op.device_put()

    def output_size(self, input_shape):
        return sum([op.output_size(input_shape) for op in self.ops])


"""
Creates an `operation` function from an operation spec.
Each operation accepts a 2D input and outputs a vector:

  op(img) -> vec

Possible operations specs are:
  * Convolutions:
    {"type":"conv", "size": (4,4) # e.g. (4,4),
     "kernel": kernel_type # either "gauss"/"random"}
  * Resampled pixels:
    {"type":"pixel", "size": (3,3)}
  * Random map:
      {"type":"random_weights", "input_size":10, "hidden_size":20}
"""
def make_operation(spec, data=None):
    optype = spec["type"]
    if optype == "pixels":
        op = PixelsOp(spec["size"])
    elif optype == "random_weights":
        op = RandWeightsOp(spec["input_size"], spec["hidden_size"])
    elif optype == "conv":
        op = ConvOp(spec["size"], spec["kernel"])
    elif optype == "gradient":
        op = GradientOp()
    elif optype == "dct":
        op = DCTOp(spec["size"])
    else:
        raise ValueError(f"Unknown input map spec: {spec['type']}")
    #  TODO: normalize all of it? such that all outputs are between (-1,1)? # 
    #_op = jax.jit(lambda img: op(img) * spec["factor"])
    return ScaleOp(spec["factor"], op)



class PixelsOp(Operation):
    def __init__(self, size):
        self.size = size
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return image.resize(img, self.size, "bilinear").reshape(-1)

    def output_size(self, input_shape):
        return self.size[0] * self.size[1]


class RandWeightsOp(Operation):
    def __init__(self, input_size, hidden_size):
        self.isize = input_size
        self.hsize = hidden_size
        self.Wih = np.random.uniform(-1, 1, (self.hsize, self.isize))
        self.bh  = np.random.uniform(-1, 1, (self.hsize,))
        self.device_put()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        Wih, bh = self.Wih, self.bh
        return Wih.dot(img.reshape(-1)) + bh

    def device_put(self):
        self.Wih = jax.device_put(self.Wih)
        self.bh  = jax.device_put(self.bh)

    def output_size(self, input_shape):
        return self.hsize


class ScaleOp(Operation):
    def __init__(self, factor, op):
        self.a = factor
        self.op = op

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return self.a * self.op(img)

    def device_put(self):
        self.op.device_put()

    def output_size(self, input_shape):
        return self.op.output_size(input_shape)


class GradientOp(Operation):
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return jnp.concatenate(jnp.gradient(img)).reshape(-1)

    def output_size(self, input_shape):
        return input_shape[0] * input_shape[1] * 2


class ConvOp(Operation):
    def __init__(self, size, kernel):
        self.size = size
        self.name = kernel
        self.kernel = get_kernel(size, kernel)[np.newaxis,np.newaxis,:,:]
        self.device_put()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        img = jnp.expand_dims(img, axis=(0,1))
        out = lax.conv(img, self.kernel, (1,1), "VALID")
        return out.reshape(-1)

    def device_put(self):
        self.kernel = jax.device_put(self.kernel)

    def output_size(self, input_shape):
        (m,n) = self.size
        return (input_shape[0]-m+1) * (input_shape[1]-n+1)


class DCTOp(Operation):
    def __init__(self, size):
        self.size = size

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return dct2(img, *self.size).reshape(-1)

    def output_size(self, input_shape):
        return self.size[0] * self.size[1]


def get_kernel(kernel_shape, kernel_type):
    if kernel_type == "mean":
        kernel = _mean_kernel(kernel_shape)
    elif kernel_type == "random":
        kernel = _random_kernel(kernel_shape)
    elif kernel_type == "gauss":
        kernel = _gauss_kernel(kernel_shape)
    else:
        raise NotImplementedError(f"Unkown kernel type `{kernel_type}`")
    return kernel


def _mean_kernel(kernel_shape):
    return np.ones(kernel_shape) / (kernel_shape[0] * kernel_shape[1])


def _random_kernel(kernel_shape):
    return np.random.uniform(size=kernel_shape, low=-1, high=1)


def _gauss_kernel(kernel_shape):
    ysize, xsize = kernel_shape
    yy = np.linspace(-ysize / 2., ysize / 2., ysize)
    xx = np.linspace(-xsize / 2., xsize / 2., xsize)
    sigma = min(kernel_shape) / 6.
    gaussian = np.exp(-(xx[:, None]**2 + yy[None, :]**2) / (2 * sigma**2))
    norm = np.sum(gaussian)  # L1-norm is 1
    gaussian = (1. / norm) * gaussian
    return gaussian
