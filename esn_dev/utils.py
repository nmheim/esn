import joblib
import numpy as np
import os.path
from os import path
from pathlib import Path
import yaml
import sys
import pyfftw
from IMED.standardizingTrans_ndim import ST_ndim_DCT, ST_ndim_FFT
from scipy.fft import set_backend

def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length
    train_seq = sequence[0:train_end]
    train_inputs  = sequence[0:train_end]
    train_targets = sequence[1:train_end+1]
    pred_targets= sequence[train_end:train_end + pred_length]
    return train_inputs, train_targets, pred_targets


def scale(x, a, b, mi=None, ma=None, inv=False):
    """Scale array 'x' to values in (a,b)"""
    if inv:
        #descale (requires mi, ma input)
        return (x-a)*(ma-mi)/(b-a)+mi
    
    if mi is None:
        mi, ma = x.min(), x.max()
    return (b-a) * (x - mi) / (ma-mi) + a

def normalize(x, mi=None, ma=None):
    """Normalize array 'x' to values in (0,1)"""
    if mi is None:
        mi, ma = x.min(), x.max()
    return (x - mi) / (ma-mi)


def _fromfile(filename):
    with open(filename, "rb") as fi:
        m = joblib.load(fi)
        #m.device_put()
    return m

def save(param_dict,tmpdir,targets,predictions,y0,h0):
    # save relevant parameters
    # and data
    
    # Make sure dir exists
    Path(f'{tmpdir}').mkdir(parents=True, exist_ok=True)
    exists = path.exists(f"{tmpdir}/esn_mses.txt")
    if not exists:
        with open(f'{tmpdir}/esn_mses.txt', "a") as f:
            # Append at end of file
            f.write(f"esn{000}\t{1e10}\t\n")
            '\n'.join(sys.argv[1:])
            f.write(f"esn{000}\t{1e10}\t\n")
            '\n'.join(sys.argv[1:])
    
    previous_esns = np.genfromtxt(f'{tmpdir}/esn_mses.txt',dtype='str')
    previous_lowest_mse = np.float64(previous_esns[-1][1])
    print(f"LOWEST MSE YET: {previous_lowest_mse}")
    
    mse = np.mean((targets - predictions)**2)
    print(f'mse is {mse}')
    folder = None
    if mse < previous_lowest_mse +1:
        esn_number = int(previous_esns[-1][0][3:]) + 1
        folder = f'{tmpdir}/esn{esn_number:03d}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        print(f'Better than previous best, saving model number {esn_number}...')
        
        with open(f'{tmpdir}/esn_mses.txt', "a") as f:
            # Append 'hello' at the end of file
            f.write(f"esn{esn_number:03d}\t{mse}\t\n")
            '\n'.join(sys.argv[1:])
            
        with open(f'{folder}/esn_arguments.yaml','w') as yaml_file:
            yaml.dump(param_dict,yaml_file)
        
        np.save(f'{folder}/y0.npy', y0)
        np.save(f'{folder}/h0.npy', h0)
        np.save(f'{folder}/predictions.npy', predictions)
        np.save(f'{folder}/targets.npy', targets)

    return folder

def preprocess(data,a,b,mi=None, ma=None, sigma=(0,2,2), eps=0.01,ST_method='DCT',workers=-1,inverse=False):
    """
    Pre-processing for data sets.
    Each data set (train, val, pred)
    Must be processed separately.
    To avoid bias, only training
    set must be used to estimate
    training parameters
    """
    
    if ST_method == 'DCT':
        ST = ST_ndim_DCT
    
    elif ST_method == 'FFT':
        ST = ST_ndim_FFT


    with set_backend(pyfftw.interfaces.scipy_fft), pyfftw.interfaces.scipy_fft.set_workers(workers):
                
        #faster if we enable cache using pyfftw
        pyfftw.interfaces.cache.enable()
        # perform standardizing transform using frequency method of your choice
        data = ST(data,sigma,eps,inverse)
    
    if (mi is None) or (ma is None):
        # Assumes training set
        mi, ma = data.min(), data.max()
        data = scale(data,a,b,mi,ma,inverse)
        return data, mi, ma
    else:
        # Appropriate for val/test set
        data = scale(data,a,b,mi,ma,inverse)
        return data