import numpy as np
from numpy.random import randint, uniform, normal
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import scipy.linalg as la
import matplotlib.pyplot as plt

# Generate a sparse hidden-to-hidden transform Whh with specified spectral radius
def sparse_esn_reservoir(Nhidden, Ncols,spectral_radius):
    # Generate the sparsity pattern: we make a regular sparse matrix
    # with Ncols nonzero columns per row.
    colidx  = randint(0,Nhidden,(Nhidden*Ncols));
    rowidx  = np.repeat(np.arange(Nhidden),Ncols)
    
    # The values are uniformly distributed between -1 and 1:
    values  = uniform(-1.0,1.0,(Nhidden*Ncols));
    #values  = normal(loc=0.0,scale=1.0,size=(Nhidden*Ncols));
    Whidden = sparse.csr_matrix( (values,(rowidx,colidx)), shape=(Nhidden,Nhidden) );
    
    # Ensure that we get the desired spectral radius 
    rho, _ = np.abs(sparse_linalg.eigs(Whidden,1))
    Whidden *= (spectral_radius/rho[0])
    
    return Whidden

# Generate hidden state sequence 
def generate_state_matrix(model,inputs,Ntrans=500):
    (Whh,Wih,bh) = model;
    (Nhidden,Ninputs,Ntrain) = (Whh.shape[0],Wih.shape[0],inputs.shape[0])
           
    H = np.zeros((Ntrain,Nhidden))    
    h = np.zeros((Nhidden,1))

    for i in range(Ntrain):
        h = np.tanh(Whh.dot(h) + Wih.dot(inputs[i]) + bh)
        H[i] = h[:,0]
    
    H0 = H[Ntrans:]
    # I2  = inputs[Ntrans-2:-2,np.newaxis]
    # I1  = inputs[Ntrans-1:-1,np.newaxis]
    I0  = inputs[Ntrans:,np.newaxis]        
    ones = np.ones((Ntrain-Ntrans,1))
    
    return np.concatenate([ones,I0,
                           H0],axis=1)

# Given a trained model W = (Wih,Whh,Who), a start internal state h0, and input y0
# predict in free-running mode for Npred steps into the future, with output feeding
# back y_n as next input:
#
#  h_{n+1} = \tanh(Whh h_n + Wih y_n)
#  y_{n+1} = Who h_{n+1}
#
def predict(model,h0,y0,Npred):
    (Whh,Wih,bh,Who) = model;
    y = y0
    h = h0.copy().reshape(-1,1)

    ys = np.zeros(Npred)    
    for i in range(Npred-1):
        h[1]  = y
        h[2:] = np.tanh(Whh.dot(h[2:])+Wih.dot(y)+bh)
        y     = Who.dot(h).reshape(1,1)
        ys[i] = y
        
    return ys, h

def lstsq_stable(H, labels):
    U, s, Vh = la.svd(H.T)
    scale = s[0]
    n = len(s[np.abs(s / scale) > 1e-5])  # Ensure condition number less than 100.000
    
    L = labels.T

    v = Vh[:n, :].T
    uh = U[:, :n].T

    wout = np.dot(np.dot(L, v) / s[:n], uh)

    return wout


def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length + 1
    train_seq = sequence[:train_end]
    inputs = train_seq[:-1]
    labels = train_seq[1:]
    pred_labels = sequence[train_end:train_end + pred_length]
    return inputs, labels, pred_labels
