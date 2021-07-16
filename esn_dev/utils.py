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


def scale(x, scale_min=-1, scale_max=1, training_min=None, training_max=None, inv=False,**kwargs):
    """Scale array 'x' to values in (scale_min,scale_max)"""
    if inv:
        #descale (requires training_min, training_max input)
        return (x-scale_min)*(training_max-training_min)/(scale_max-scale_min)+training_min
    
    if training_min is None:
        training_min, training_max = x.min(), x.max()
    return (scale_max-scale_min) * (x - training_min) / (training_max-training_min) + scale_min

def normalize(x, training_min=None, training_max=None,**kwargs):
    """Normalize array 'x' to values in (0,1)"""
    if training_min is None:
        training_min, training_max = x.min(), x.max()
    return (x - training_min) / (training_max-training_min)


def _fromfile(filename):
    with open(filename, "rb") as fi:
        m = joblib.load(fi)
        #m.device_put()
    return m

def save(targets, predictions,dict_of_arrays,param_dict=None,save_condition='if_better',savedir='tmp'):
    # save relevant parameters
    # and data, and making folder for plotting output
    
    # Make sure dir exists
    Path(f'{savedir}').mkdir(parents=True, exist_ok=True)
    exists = path.exists(f"{savedir}/esn_mses.txt")
    if not exists:
        with open(f'{savedir}/esn_mses.txt', "a") as f:
            # Append at end of file
            f.write(f"esn{000}\t{1e10}\t\n")
            '\n'.join(sys.argv[1:])
            f.write(f"esn{000}\t{1e10}\t\n")
            '\n'.join(sys.argv[1:])
    
    previous_esns = np.genfromtxt(f'{savedir}/esn_mses.txt',dtype='str')
    previous_lowest_mse = np.float64(previous_esns[-1][1])
    
    mse = np.mean((targets - predictions)**2)
    print(f'mse is {mse}')
    
    if save_condition == 'if_better':
        #optional save if better than previous MSE
        save_condition = (mse < previous_lowest_mse)
        if save_condition:
            print('Saving model, better than previous.')
        else:
            print('Not saving model. Is worse.')
    elif save_condition == 'always':
        save_condition = True
        print('Always saving model')
        
    elif save_condition == 'never':
        save_condition = False
        prnit('Never saving model.')
    
    folder = None
    if save_condition:
        esn_number = int(previous_esns[-1][0][3:]) + 1
        folder = f'{savedir}/esn{esn_number:03d}'
        print(f'Saving at {savedir}/esn{esn_number:03d}')

        if not os.path.exists(folder):
            os.makedirs(folder)
        
        with open(f'{savedir}/esn_mses.txt', "a") as f:
            # Append 'hello' at the end of file
            f.write(f"esn{esn_number:03d}\t{mse}\t\n")
            '\n'.join(sys.argv[1:])

        with open(f'{folder}/esn_arguments.yaml','w') as yaml_file:
            yaml.dump((param_dict),yaml_file)
        import pickle
        with open(f'{folder}/esn_arguments.pkl','wb') as pickle_file:
            pickle.dump((param_dict),pickle_file)
        """
        np.save(f'{folder}/y0.npy', y0)
        np.save(f'{folder}/h0.npy', h0)
        np.save(f'{folder}/predictions.npy', predictions)
        np.save(f'{folder}/targets.npy', targets)"""
        for arr_name, arr in dict_of_arrays.items():
            print(arr_name)
            np.save(f'{folder}/{arr_name}.npy',arr)
            
    return folder

def preprocess(data,scale_min=-1,scale_max=1,training_min=None, training_max=None,
               sigma=(0,2,2), eps=0.01,ST_method='DCT',cpus_to_use =-1,inverse=False,**kwargs):
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


    with set_backend(pyfftw.interfaces.scipy_fft), pyfftw.interfaces.scipy_fft.set_workers(cpus_to_use):
                
        #faster if we enable cache using pyfftw
        pyfftw.interfaces.cache.enable()
        # perform standardizing transform using frequency method of your choice
        data = ST(data,sigma,eps,inverse)
    
    if (training_min is None) or (training_max is None):
        # Assumes training set
        training_min, training_max = data.min(), data.max()
        data = scale(data,scale_min,scale_max,training_min,training_max,inverse)
        return data, training_min, training_max
    else:
        # Appropriate for val/test set
        data = scale(data,scale_min,scale_max,training_min,training_max,inverse)
        return data
    
    
def score(predictions, targets):
    """
    MSE of predictions and targets.
    If targets and predictions are standardizing transformed,
    the MSE is equal to the IMED    
    
    Returns:
        MSE: (float). Single MSE over all time steps
    """
    
    MSE = np.square(predictions-targets).mean()
    
    return MSE
    
def score_over_time(predictions, targets):
    """
    MSE of predictions and targets.
    If targets and predictions are standardizing transformed,
    the MSE is equal to the IMED.    
    
    Params:
    ...    assumes predictions, targets have shape (T,W,H)
    
    Returns:
        MSEs: (ndarray). MSE for each time step
    """
    
    MSEs = np.square(predictions-targets).mean(axis=1).mean(axis=1)
    
    return MSEs