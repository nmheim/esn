import numpy as np
from esn_dev.hidden import evolve_hidden_state, dimension_reduce
from sklearn.decomposition import KernelPCA, PCA

def lin_readout(h,Who,pca_object=None):
    """
    Given a trained output matrix Who, transform
    hidden state h to output y
          
    Params:
        h: single hidden state or state matrix (H)
        Who: fitted output matrix such that y=Who.h
        pca_object: already-fitted PCA transform if using
                    if None: No dimension reduction.
    Returns: 
        y: output
    """
    
    # dimension-reduce
    if pca_object is not None:
        h = dimension_reduce(h,pca_object)
    
    return Who.dot(h)


def predict(model, y0, h0, pca_object=None, Npred=300,dtype=None):
    """
    Given a trained model = (Wih,Whh,bh,Who), a start internal state h0, and input
    y0 predict in free-running mode for Npred steps into the future, with
    output feeding back y_n as next input:

      h_{n+1} = \tanh(Whh h_n + Wih y_n + bh)
      y_{n+1} = Who h_{n+1}
      
    Params:
        model: complete trained model (map_ih,Whh,bh,Who)
        y0: last 2D target from training set
        h0: last hidden state from training mode
        Npred: Number of free-running predictions to make
        Who: fitted output matrix such that y=Who.h
        pca_object: already-fitted PCA transform if using
                    if None: No dimension reduction.
    Returns: 
        Y: sequence of 2D outputs (Npred,y.shape[0].y.shape[1])
    """
    (map_ih,Whh,bh,Who) = model
    #infer number of stacked inputs and number of bias parameters in h0
    
    assert(y0.ndim == 2)
    
    Y  = np.zeros([Npred,y0.size],dtype=dtype)
    h  = evolve_hidden_state(model, y0, h0,mode='predict')
    for t in range(0,Npred):
        Y[t] = lin_readout(h,Who,pca_object)
        y    = Y[t].reshape(y0.shape)
        h    = evolve_hidden_state(model,y,h,mode='predict')

    Y = Y.reshape((Npred,y0.shape[0],y0.shape[1]))

    return Y

