import pytest
import joblib
import numpy as np
from jax.profiler import start_trace
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from esn.input_map import InputMap
from esn.utils import split_train_label_pred
from esn.toydata import gauss2d_sequence, mackey2d_sequence
import esn.sparse_esn as se
import pyfftw
from scipy.fft import set_backend

from IMED.standardizingTrans_ndim import ST_ndim_DCT
from time import time
import gc
import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
            
def print_globals():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
                         key= lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    

def sparse_esn_2d_train_pred(tmpdir, data, specs,
                             spectral_radius=1.5,
                             density=0.01,
                             Ntrans=500,        # Number of transient initial steps before training
                             Ntrain=2500,       # Number of steps to train on
                             Npred=500,         # Number of steps for free-running prediction
                             mse_threshold=1e-5,
                             plot_prediction=False,eps=1e-5,sigma=(1,5,5)):
    print(f'Using spectral radius {spectral_radius}')
    np.random.seed(1)
    N = Ntrain + Npred + 1
    assert data.ndim == 3
    assert data.shape[0] >= N
    
    """with set_backend(pyfftw.interfaces.scipy_fft), pyfftw.interfaces.scipy_fft.set_workers(-1):
        print('USING NO TEMPORAL CORRELATION')
        #faster if we enable cache using pyfftw
        pyfftw.interfaces.cache.enable()
        # perform standardizing transform using frequency method of your choice
        data = ST_ndim_DCT(data,sigma=(0,sigma[1],sigma[2]),eps=eps,inverse=False)"""
   
    
    # prepare data
    inputs, labels, pred_labels = split_train_label_pred(data,Ntrain,Npred)
    pred_labels_orig = np.copy(pred_labels)
    
    #transform input - labels should be independent from input
    print(f'Using sigma: {sigma}, eps: {eps}')
    with set_backend(pyfftw.interfaces.scipy_fft), pyfftw.interfaces.scipy_fft.set_workers(-1):
        #faster if we enable cache using pyfftw
        pyfftw.interfaces.cache.enable()
        # perform standardizing transform using frequency method of your choice
        inputs = ST_ndim_DCT(inputs,sigma=sigma,eps=eps,inverse=False)
        labels = ST_ndim_DCT(labels,sigma=sigma,eps=eps,inverse=False)
        pred_labels = ST_ndim_DCT(pred_labels,sigma=sigma,eps=eps,inverse=False)

    img_shape = inputs.shape[1:]
    

    # build esn
    start = time()
    map_ih = InputMap(specs)
    hidden_size = map_ih.output_size(img_shape)
    print("Hidden size: ", hidden_size)
    esn = se.esncell(map_ih, hidden_size, spectral_radius=spectral_radius, density=density)
    esn[-1].block_until_ready()
    end = time()
    print(f'building took {end-start:.2f}')
        
    start = time()
    # compute training states
    print(f'inputs have shape {inputs.shape}')
    H = se.augmented_state_matrix(esn, inputs, Ntrans)
    print(f'H has shape {H.shape}')
    H.block_until_ready()
    
    end = time()
    print(f'state harvesting took {end-start:.2f}')
     
    
    # compute last layer without imed
    _labels = labels.reshape(inputs.shape[0], -1)
    print(f'_labels[Ntrans:] has shape {_labels[Ntrans:].shape}')
    start = time()
    model = se.train(esn, H, _labels[Ntrans:])
    model[-1].block_until_ready()
    end = time()
    print(f'lstsq optimization took {end-start:.2f}')
    
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                     key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    
    # predict
    start = time()
    y0, h0 = labels[-1], H[-1]
    del H
    gc.collect()
    print(f'whoVh takes up {model[-1].nbytes*1e-9}GB with shape {model[-1].shape}, dtype {model[-1].dtype}')

    #(y,h), (ys,hs) = se.predict(model, y0, h0, Npred)
    #(y,h), (ys,hs) = se.predict_scipy(model, y0, h0, Npred)
    #ys = se.predict_scipy(model, y0, h0, Npred,return_H=False)
    ys = se.predict_scipy_stack(model, y0, h0, Npred,return_H=False)

    jnp.asarray(ys).block_until_ready()
    end = time()
    print(f'predicting took {end-start:.2f}')
    
    
    print(f'ys has shape {ys.shape}')
    # predict with warump of Ntrain frames
    #_, (wys,_) = se.warmup_predict(model, labels[-Ntrans:], Npred)

      
    mse = jnp.mean((ys - pred_labels)**2)
    #w_mse = jnp.mean((wys[25] - pred_labels[25])**2)
    print("MSE before inverse ST:  ", mse)
    
    
    with set_backend(pyfftw.interfaces.scipy_fft),pyfftw.interfaces.scipy_fft.set_workers(-1):
        #faster if we enable cache using pyfftw
        pyfftw.interfaces.cache.enable()
        # perform standardizing transform using frequency method of your choice
        ys = ST_ndim_DCT(ys,sigma=sigma,eps=eps,inverse=True)
        pred_labels = ST_ndim_DCT(pred_labels,sigma=sigma,eps=eps,inverse=True)
        #wys = ST_ndim_DCT(wys,sigma=sigma,eps=eps,inverse=True)
    
    mse = jnp.mean((ys - pred_labels)**2)
    #w_mse = jnp.mean((wys[25] - pred_labels[25])**2)
    print("MSE after inverse ST:  ", mse)
    
    mse = jnp.mean((ys - pred_labels_orig)**2)
    #w_mse = jnp.mean((wys[25] - pred_labels[25])**2)
    print("MSE orig after inverse ST:  ", mse)
    
    #assert mse < mse_threshold
    #assert w_mse < mse_threshold
    #assert jnp.isclose(mse, w_mse, atol=1e-3)
    if plot_prediction:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(pred_labels.sum(axis=0))
        ax[0].set_title("Truth")
        ax[1].imshow(ys.sum(axis=0))
        ax[1].set_title("Pred.")

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(pred_labels[-1])
        ax[0].set_title("Truth")
        ax[1].imshow(ys[-1])
        ax[1].set_title("Pred.")

        from IPython.display import HTML
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        def animate_this(anim_data):
            def init():
                im.set_data(anim_data[0,:,:])
                return (im,)

            # animation function. This is called sequentially
            def animate(i):
                data_slice = anim_data[i,:,:]
                im.set_data(data_slice)
                return (im,)
            fig, ax = plt.subplots()
            im = ax.imshow(anim_data[0,:,:])

            # call the animator. blit=True means only re-draw the parts that have changed.
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=anim_data.shape[0], interval=20, blit=True)
            return anim

        ys_anim = animate_this(ys)
        pred_labels_orig_anim = animate_this(pred_labels_orig)
        ys_anim.save(f'{tmpdir}/ys_anim.gif',writer=animation.PillowWriter(fps=24))
        pred_labels_orig_anim.save(f'{tmpdir}/pred_labels_orig_anim.gif',writer=animation.PillowWriter(fps=24))

    
"""
    with open(tmpdir / "esn.pkl", "wb") as fi:
        joblib.dump(model, fi)
    pkl_model = se.load_model(tmpdir / "esn.pkl")
    _, (pkl_ys,_) = se.predict(pkl_model, y0, h0, Npred)
    print(ys[0])
    print(pkl_ys)
    assert jnp.all(jnp.isclose(pkl_ys, ys))
"""


def test_sparse_esn_lissajous(tmpdir):
    input_shape = (100,100)
    input_size  = input_shape[0] * input_shape[1]

    from esn.utils import scale
    data = gauss2d_sequence(size=input_shape)
    data = scale(data, -1, 1)
    specs = [
        {"type":"pixels", "size":input_shape, "factor": 3.},
        {"type":"conv", "size":(3,3),   "kernel":"gauss",  "factor": 2.},
        {"type":"conv", "size":(5,5),   "kernel":"gauss",  "factor": 2.},
        {"type":"conv", "size":(7,7),   "kernel":"gauss",  "factor": 2.},
        {"type":"conv", "size":(9,9),   "kernel":"gauss",  "factor": 2.},
        {"type":"conv", "size":(3,3),   "kernel":"random", "factor": 2.},
        {"type":"conv", "size":(5,5),   "kernel":"random", "factor": 2.},
        {"type":"conv", "size":(7,7),   "kernel":"random", "factor": 2.},
        {"type":"conv", "size":(9,9),   "kernel":"random", "factor": 2.},
        {"type":"gradient", "factor": 2.},
        {"type":"dct", "size":(15,15), "factor": 0.1},
        {"type":"random_weights", "input_size":input_size, "hidden_size":3500, "factor": 1.},
    ]

    sparse_esn_2d_train_pred(tmpdir, data, specs,
        plot_prediction=True, mse_threshold=1e-15,
        spectral_radius=2.0, density=10,
        Ntrain=2500, Npred=300, Ntrans=500,eps=1e-7,sigma=(0.,1,1))


def test_sparse_esn_chaotic(tmpdir):
    print('Running chaotic...')
    input_shape = (30,30)
    input_size  = input_shape[0] * input_shape[1]

    data = mackey2d_sequence(size=input_shape)
    
    specs = [
        #{"type":"pixels", "size":input_shape, "factor": 1.},
        {"type":"conv", "size":(3,3),   "kernel":"gauss",  "factor": 1.},        
        {"type":"conv", "size":(5,5),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(7,7),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(9,9),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(11,11),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(13,13),   "kernel":"gauss",  "factor": 1.},
        {"type":"gradient", "factor": 1.},
        #{"type":"dct", "size":(5,5), "factor": 1},
        #{"type":"dct", "size":(13,13), "factor": 1},
        #{"type":"random_weights", "input_size":input_size, "hidden_size":15000, "factor": 1.},
    ]
    

    
    sparse_esn_2d_train_pred(tmpdir, data, specs,
        plot_prediction=True, mse_threshold=1e-2,
        spectral_radius=1.4, density=49,
        Ntrain=3000, Npred=300, Ntrans=500,eps=1e-5,sigma=(0.05,0.5,0.5))



if __name__ == "__main__":
    #test_sparse_esn_lissajous("tmp")
    test_sparse_esn_chaotic("tmp")

