from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax import image

from esn.dct import dct2
from esn.utils import _fromfile, normalize


def make_operation(spec):
    """
    Creates an `Operation` from an operation spec.
    Each operation accepts a 2D input and outputs a vector:

      op(img) -> vec

    Possible operations specs are e.g.:

      * Resampled pixels:
        {"type":"pixel", "size": (3,3), "factor":1.}

      * Random projection:
          {"type":"random_weights", "input_size":10, "hidden_size":20,
           "factor": 1.}

      * Convolutions:
        {"type":"conv", "size": (4,4), # kernel size
         "kernel": kernel_type,        # either "gauss"/"random"
         "factor": 1.}

      * Gradient:
        {"type":"gradient", "factor": 1.}

      * Discrete Cosine Transform:
        {"type":"dct", "size": (n,n)  # pick first n coefficients in each dimension,
         "factor":0.1}

    Every operation spec must contain at least a 'type' determining the kind of
    operation and a 'factor' that is applied to the operation before outputing
    the result (realized through a 'ScaleOp')
    """
    optype = spec["type"]
    if optype == "pixels":
        op = PixelsOp(spec["size"])
    elif optype == "random_weights":
        op = RandWeightsOp(spec["input_size"], spec["hidden_size"])
    elif optype == "conv":
        op = ConvOp(spec["size"], spec["kernel"])
    elif optype == "gradient":
        op = GradientOp()
    elif optype == "vorticity":
        op = VorticityOp()        
    elif optype == "dct":
        op = DCTOp(spec["size"])
    else:
        raise ValueError(f"Unknown input map spec: {spec['type']}")
    #  TODO: normalize all of it? such that all outputs are between (-1,1)? # 
    #_op = jax.jit(lambda img: op(img) * spec["factor"])
    return ScaleOp(spec["factor"], op)


def rescale(mapih, factors):
    ops = [ScaleOp(f,op.op) for (f,op) in zip(factors, mapih.ops)]
    return InputMap(ops)


class Operation:

    @classmethod
    def fromfile(cls, filename):
        return _fromfile(filename)

    def device_put(self):
        pass

    @classmethod
    def fromspec(self, spec):
        return make_operation(spec)


class InputMap(Operation):
    """
    A callable object, that can be called with a 2D input image.  The
    `InputMap` is composed of a number of `Operation`s.  Each `Operation` again
    takes an image as input and outputs a vector.  Possible operations include
    convolutions, random maps, resize, etc. For a full list of operations and
    their specifications see `make_operation`.

    Params:
        specs: list of dicts with that each specify an `operation`

    Returns:
        A function that can be called with a 2D array and that outputs
        a 1D array (concatenated output of each op).
    """
    def __init__(self, xs):
        if isinstance(xs[0], dict):
            self.ops = [make_operation(spec) for spec in xs]
        elif isinstance(xs[0], ScaleOp):
            self.ops = xs
        else:
            raise ValueError("'xs' must either be a list of 'ScaleOp's or 'dict's!")

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return jnp.concatenate([op(img) for op in self.ops], axis=0)

    def device_put(self):
        for op in self.ops: op.device_put()

    def output_size(self, input_shape):
        return sum([op.output_size(input_shape) for op in self.ops])


class PixelsOp(Operation):
    def __init__(self, size):
        self.size = size
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return image.resize(img, self.size, "bilinear").reshape(-1)

    def output_size(self, input_shape):
        return self.size[0] * self.size[1]

    def output_shape(self, input_shape):
        return (self.size[0], self.size[1])



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

    def output_shape(self, input_shape):
        raise NotImplementedError


class ScaleOp(Operation):
    def __init__(self, factor, op):
        self.factor = factor
        self.op = op

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return self.factor * self.op(img)

    def device_put(self):
        self.op.device_put()

    def output_size(self, input_shape):
        return self.op.output_size(input_shape)

    def output_shape(self, input_shape):
        return self.op.output_shape(input_shape)


class GradientOp(Operation):
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        x = jnp.gradient(img)
        x = jnp.concatenate(x).reshape(-1)
        return normalize(x)*2-1

    def output_size(self, input_shape):
        s = self.output_shape(input_shape)
        return s[0] * s[1]

    def output_shape(self, input_shape):
        return (input_shape[0]*2, input_shape[1])

class VorticityOp(Operation):
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        u, v = jnp.gradient(img)
        ux, uy = jnp.gradient(u)
        vx, vy = jnp.gradient(v)
        vorticity = ux - vy
        return vorticity.reshape(-1)

    def output_size(self, input_shape):
        s = self.output_shape(input_shape)
        return s[0] * s[1]

    def output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])
    

class ConvOp(Operation):
    def __init__(self, size, kernel, padding="SAME"):
        self.size = size
        self.name = kernel
        self.kernel = get_kernel(size, kernel)[np.newaxis,np.newaxis,:,:]
        self.padding = padding
        self.device_put()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        img = jnp.expand_dims(img, axis=(0,1))
        out = lax.conv(img, self.kernel, (1,1), self.padding)
        return out.reshape(-1)

    def device_put(self):
        self.kernel = jax.device_put(self.kernel)

    def output_size(self, input_shape):
        s = self.output_shape(input_shape)
        return s[0] * s[1]

    def output_shape(self, input_shape):
        p = self.padding
        if p == "SAME":
            return input_shape
        elif p == "VALID":
            (m,n) = self.size
            return (input_shape[0]-m+1, input_shape[1]-n+1)
        else:
            raise ValueError(f"'padding' must be either 'VALID' or 'SAME' - got: {p}")


class DCTOp(Operation):
    def __init__(self, size):
        self.size = size

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, img):
        return dct2(img, *self.size).reshape(-1)

    def output_size(self, input_shape):
        return self.size[0] * self.size[1]

    def output_shape(self, input_shape):
        return self.size


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
