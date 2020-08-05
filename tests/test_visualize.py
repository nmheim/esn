import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
from esn.input_map import InputMap
import esn.sparse_esn as se
from esn.visualize import plot_input_map

mpl.use('Agg')

def test_plot_input_map():
    IMG_SHAPE = (10,10)
    RAND_SPEC = {"type":"random_weights",
                "input_size":IMG_SHAPE[0]*IMG_SHAPE[1],
                "hidden_size":20,
                "factor":0.5}
    
    PIXEL_SPEC = {"type":"pixels", "size":(3,3), "factor": 0.5}
    
    CONV_SPEC = {"type":"conv", "size":(2,2), "kernel":"gauss", "factor": 0.5}
    
    GRAD_SPEC = {"type":"gradient", "factor": 0.5}
    
    DCT_SPEC  = {"type":"dct", "size":(3,3), "factor": 0.01}
    
    SPECS = [RAND_SPEC, CONV_SPEC, PIXEL_SPEC, GRAD_SPEC, DCT_SPEC]
    
    map_ih = InputMap(SPECS)
    esn = se.esncell(map_ih, map_ih.output_size(IMG_SHAPE))
    img = jax.device_put(np.random.uniform(size=IMG_SHAPE))
    state = jax.device_put(np.random.uniform(size=(map_ih.output_size(IMG_SHAPE),))) 
    
    plot_input_map(esn, img, state)
    assert True
