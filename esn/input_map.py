import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
#from jax import image TODO: why does this not work???

from esn.dct import dct2


"""
Creates a function `input_map` that can be called with a 2D input image.
The `input_map` is composed of a number of `operation`s.
Each `operation` again takes an image as input and outputs a vector.
Possible operations include convolutions, random maps, resize, etc. For a full
list of operations and their specifications see `make_operation`.

Params:
    specs: list of dicts with that each specify an `operation`

Returns:
    A function that can be called with a 2D array and that outputs
    a 1D array (concatenated output of each op).
"""
def make_input_map(specs):
    ops = [make_operation(spec) for spec in specs]
    return lambda img: jnp.concatenate([op(img) for op in ops], axis=0)


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
def make_operation(spec):
    optype = spec["type"]
    if optype == "pixels":
        operation = make_pixels_op(spec)
    elif optype == "random_weights":
        operation = make_random_weighs_op(spec)
    elif optype == "conv":
        operation = make_conv_op(spec)
    elif optype == "gradient":
        operation = lambda img: jnp.concatenate(jnp.gradient(img)).reshape(-1)
    elif optype == "dct":
        operation = make_dct_op(spec)
    else:
        raise ValueError(f"Unknown input map spec: {spec['type']}")
    #  TODO: normalize all of it? such that all outputs are between (-1,1)? # 
    return operation


def map_output_size(input_map_specs, input_shape):
    return sum(map(lambda s: op_output_size(s,input_shape), input_map_specs))


def op_output_size(spec, input_shape):
    optype = spec["type"]
    if optype == "conv":
        (m,n) = spec["size"]
        size = (input_shape[0]-m+1) * (input_shape[1]-n+1)
    elif optype == "random_weights":
        size = spec["hidden_size"]
    elif optype == "gradient":
        size = input_shape[0] * input_shape[1] * 2  # specor 2d pictures
    elif optype == "compose":
        size = sum(map(output_size, spec["operations"]))
    elif optype in ["pixels", "dct"]:
        shape = spec["size"]
        size = shape[0] * shape[1]
    else:
        raise ValueError(f"Unknown input map spec: {spec['type']}")
    return size


def make_pixels_op(spec):
    #return lambda img: image.resize(img, spec["size"], "bilinear").reshape(-1)
    return lambda img: img.reshape(-1)


def make_random_weighs_op(spec):
    isize, hsize = spec["input_size"], spec["hidden_size"]
    Wih = np.random.uniform(-1, 1, (hsize, isize))
    Wih = jax.device_put(Wih)
    bh  = np.random.uniform(-1, 1, (hsize,))
    bh  = jax.device_put(bh)
    return lambda img: Wih.dot(img.reshape(-1)) + bh


def make_conv_op(spec):
    kernel = get_kernel(spec["size"], spec["kernel"])
    kernel = jax.device_put(kernel[np.newaxis,np.newaxis,:,:])
    def operation(img):
        img = jnp.expand_dims(img, axis=(0,1))
        out = lax.conv(img, kernel, (1,1), "VALID")
        return out.reshape(-1)
    return operation


def make_dct_op(spec):
    #@jax.custom_vjp
    #def _dct2(jax_img, n, m):
    #    x = np.asarray(jax_img)
    #    x = dct2(x, n, m).reshape(-1)
    #    return jax.device_put(x)
    #def _dct2_rev(primals, tangents):
    #    raise
    #    return primals, tangents
    #def _dct2_fwd(primals, tangens):
    #    raise
    #    return primals, tangents
    #_dct2.defvjp(_dct2_fwd, _dct2_rev)
    return lambda img: dct2(img, *spec["size"]).reshape(-1)


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
