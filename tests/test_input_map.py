import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from esn.input_map import (make_operation, make_input_map, op_output_size,
                           map_output_size)


IMG_SHAPE = (5,5)

RAND_SPEC = {"type":"random_weights",
            "input_size":IMG_SHAPE[0]*IMG_SHAPE[1],
            "hidden_size":20,
            "factor":0.5}

PIXEL_SPEC = {"type":"pixels", "size":(5,5), "factor": 0.5}

CONV_SPEC = {"type":"conv", "size":(2,2), "kernel":"gauss", "factor": 0.5}

GRAD_SPEC = {"type":"gradient", "factor": 0.5}

DCT_SPEC  = {"type":"dct", "size":(3,3), "factor": 0.5}

SPECS = [RAND_SPEC, CONV_SPEC, PIXEL_SPEC, GRAD_SPEC, DCT_SPEC]

def test_rand_operation():
    spec = RAND_SPEC
    img = jax.device_put(np.random.uniform(size=IMG_SHAPE))
    op = make_operation(spec)
    assert op(img).shape == (op_output_size(spec, IMG_SHAPE),)
    # test that "random_weights" also works for vectors
    vec = img.reshape(-1)
    assert op(vec).shape == (op_output_size(RAND_SPEC, IMG_SHAPE),)

def test_pixel_operation():
    spec = PIXEL_SPEC
    img = jax.device_put(np.random.uniform(size=IMG_SHAPE))
    op = make_operation(spec)
    assert op(img).shape == (op_output_size(spec, IMG_SHAPE),)

def test_conv_operation():
    spec = CONV_SPEC
    img = jax.device_put(np.random.uniform(size=IMG_SHAPE))
    op = make_operation(spec)
    assert op(img).shape == (op_output_size(spec, IMG_SHAPE),)

def test_grad_operation():
    spec = GRAD_SPEC
    img = jax.device_put(np.random.uniform(size=IMG_SHAPE))
    op = make_operation(spec)
    assert op(img).shape == (op_output_size(spec, IMG_SHAPE),)

def test_dct_operation():
    spec = DCT_SPEC
    img = jax.device_put(np.random.uniform(size=IMG_SHAPE))
    op = make_operation(spec)
    assert op(img).shape == (op_output_size(spec, IMG_SHAPE),)

def test_input_map():
    img = jax.device_put(np.random.uniform(size=(IMG_SHAPE)))
    map_ih = make_input_map(SPECS)
    assert map_ih(img).shape == (map_output_size(SPECS, IMG_SHAPE),)

if __name__ == "__main__":
    test_make_operation()
    test_input_map()
