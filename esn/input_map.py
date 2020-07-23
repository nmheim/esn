import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
#from jax import image TODO: why does this not work???


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


def make_pixel_op(spec):
    return lambda img: image.resize(img, spec["size"], "bilinear").reshape(-1)


def make_random_weighs_op(spec):
    idim, hdim = spec["input_dim"], spec["hidden_dim"]
    Wih = np.random.uniform(-1, 1, (hdim, idim))
    Wih = jax.device_put(Wih)
    bh  = np.random.uniform(-1, 1, (hdim,))
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


def make_operation(spec):
    if spec["type"] == "pixels":
        operation = make_pixel_op(spec)
    elif spec["type"] == "random_weights":
        operation = make_random_weighs_op(spec)
    elif spec["type"] == "conv":
        operation = make_conv_op(spec)
    else:
        raise ValueError(f"Unknown input map spec: {spec['type']}")
    #  TODO: normalize all of it? such that all outputs are between (-1,1)? # 
    return operation


def make_input_map(specs):
    ops = [make_operation(spec) for spec in specs]
    return lambda img: jnp.concatenate([op(img) for op in ops], axis=0)


def op_output_size(spec, input_shape):
    if spec["type"] == "conv":
        if spec["mode"] == "valid":
            (m,n) = spec["size"]
            size = (input_shape[0]-m+1) * (input_shape[1]-n+1)
        elif spec["mode"] == "same":
            size = input_shape[0] * input_shape[1]
    elif spec["type"] == "random_weights":
        size = spec["hidden_dim"]
    elif spec["type"] == "gradient":
        size = input_shape[0] * input_shape[1] * 2  # specor 2d pictures
    elif spec["type"] == "compose":
        size = sum(map(output_size, spec["operations"]))
    elif spec["type"] == "pixels":
        size  = shape[0] * shape[1]
    return size
            

def map_output_size(input_map_specs, input_shape):
    return sum(map(lambda s: op_output_size(s,input_shape), input_map_specs))
