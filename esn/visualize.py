import numpy as np
import matplotlib.pyplot as plt
from esn.jaxsparse import sp_dot
from esn.input_map import RandWeightsOp

NRPLOTS_TO_FIGSIZE = {
    1:  (1, 1),
    2:  (1, 2),
    3:  (2, 2),
    4:  (2, 2),
    5:  (2, 3),
    6:  (2, 3),
    7:  (2, 4),
    8:  (2, 4),
    9:  (3, 3),
    10: (2, 5),
    11: (3, 4),
    12: (3, 4),
    13: (3, 5),
    14: (3, 5),
    15: (3, 5),
    16: (4, 4),
    17: (3, 6),
    18: (3, 6),
}
 
def vec_to_rect(vec):
    size = int(np.ceil(vec.shape[0]**.5))
    shape = (size, size)
    pad = np.zeros(size * size - vec.shape[0])
    pad[:] = np.nan
    rect = np.concatenate([vec, pad], axis=0).reshape(shape)
    return rect

def plot_input_map(esn, img, h0):
    # new state
    h1 = np.array(esn.apply(img.reshape(1,*img.shape), h0)[0])
    # all input maps concat'ed
    ih = np.array(esn.map_ih(img))
    # hidden to hidden
    hh = np.array(sp_dot(esn.Whh, h0, esn.hidden_size))

    img = np.array(img)
    h0 = np.array(h0)

    nr_plots = len(esn.map_ih.ops) + 5
    height, width = NRPLOTS_TO_FIGSIZE[nr_plots]
    fig, ax = plt.subplots(height, width, figsize=(10, 10))
    ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    im = ax[0].imshow(img)
    ax[0].set_title("img")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(vec_to_rect(h0))
    ax[1].set_title("h0")
    plt.colorbar(im, ax=ax[1])

    # individual input maps
    vs = [np.array(op(img)) for op in esn.map_ih.ops]
    for i in range(nr_plots - 5):
        op = esn.map_ih.ops[i]
        v  = np.array(op(img))
        axi = ax[i+2]

        # assumes that all ops are ScaleOps and that we actually want the op
        # that is wrapped
        if type(op.op) == RandWeightsOp:
            arr = vec_to_rect(v)
        else:
            arr = v.reshape(op.output_shape(img.shape))

        im = axi.imshow(arr)
        axi.set_title(f"{type(op.op).__name__}(img) #map: {i}")
        plt.colorbar(im, ax=axi)

    im = ax[-3].imshow(vec_to_rect(hh))
    ax[-3].set_title("Whh * h0")
    plt.colorbar(im, ax=ax[-3])

    im = ax[-2].imshow(vec_to_rect(hh + ih))
    ax[-2].set_title("Whh * h0 + map_ih(img)")
    plt.colorbar(im, ax=ax[-2])

    im = ax[-1].imshow(vec_to_rect(h1))
    ax[-1].set_title("tanh(Whh * h0 + map_ih(img) + bh)")
    plt.colorbar(im, ax=ax[-1])
    return fig, ax

if __name__ == "__main__":
    import jax
    from esn.input_map import InputMap
    from esn.sparse_esn import SparseESN

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
    esn = SparseESN(map_ih, map_ih.output_size(IMG_SHAPE))
    img = jax.device_put(np.random.uniform(size=IMG_SHAPE))
    state = jax.device_put(np.random.uniform(size=(map_ih.output_size(IMG_SHAPE),))) 

    plot_input_map(esn, img, state)
