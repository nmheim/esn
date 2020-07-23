import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from esn.input_map import (make_operation, make_input_map, op_output_size,
                           map_output_size)

IMG_SHAPE = (5,5)
RAND_SPEC = {"type":"random_weights",
            "input_dim":IMG_SHAPE[0]*IMG_SHAPE[1],
            "hidden_dim":20}
PIXEL_SPEC = {"type":"pixels", "size":(2,2)}
CONV_SPEC = {"type":"conv", "size":(2,2), "kernel":"gauss", "mode":"valid"}


def test_make_operation():
    img = np.random.uniform(size=IMG_SHAPE)

    op = make_operation(RAND_SPEC)
    assert op(img).shape == (op_output_size(RAND_SPEC, IMG_SHAPE),)

    # op = make_mapih_operation(spec)
    # assert op(img).shape == (2*2,)
    
    op = make_operation(CONV_SPEC)
    assert op(img).shape == (op_output_size(CONV_SPEC, IMG_SHAPE),)


def test_input_map():
    img = np.random.uniform(size=(5,5))
    specs = [RAND_SPEC, CONV_SPEC]
    map_ih = make_input_map(specs)
    assert map_ih(img).shape == (map_output_size(specs, IMG_SHAPE),)

if __name__ == "__main__":
    test_make_operation()
    test_input_map()
