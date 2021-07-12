import numpy as np
import gc #garbage collector
import scipy

def lstsq_stable(H, labels, thresh=1e-4):
    """    if labels.ndim != 2:
        raise ValueError("Labels must have shape (time, features)")
    
    print('Doing svd')
    U, s, Vh = scipy.linalg.svd(H.T, full_matrices=False)
    scale = s[0]
    n = np.sum(np.abs(s/scale) > thresh)  # Ensure condition number less than 1/thresh
    
    L = labels.T
    v = Vh[:n, :].T
    uh = U[:, :n].T
    
    print(f'H takes up {H.nbytes*1e-9}GB with shape {H.shape}, dtype {H.dtype}')
    print(f'U takes up {U.nbytes*1e-9}GB with shape {U.shape}, dtype {U.dtype}')
    print(f'Vh takes up {Vh.nbytes*1e-9}GB with shape {Vh.shape}, dtype {Vh.dtype}')
    #del U #H and U are both of size 
    del H
    gc.collect()
    print('deleted H, U')
    Who =  np.dot(np.dot(L, v) / s[:n], uh) 
    #alpha = s[0]*0.001
    #print(f'Tikhonov with parameter {alpha}, largest Singualar value: {s[0]:.3e}')
    #Who =  np.dot(s*np.dot(L, Vh.T) / (alpha +s**2), U.T)"""
    
    if labels.ndim != 2:
        raise ValueError("Labels must have shape (time, features)")
    
    print('Doing svd')
    U, s, Vh = scipy.linalg.svd(H, full_matrices=False)
    scale = s[0]
    n = np.sum(np.abs(s/scale) > thresh)  # Ensure condition number less than 1/thresh
    
    L = labels
    v = Vh[:n, :]
    uh = U[:, :n]
    
    print(f'H takes up {H.nbytes*1e-9}GB with shape {H.shape}, dtype {H.dtype}')
    print(f'U takes up {U.nbytes*1e-9}GB with shape {U.shape}, dtype {U.dtype}')
    print(f'Vh takes up {Vh.nbytes*1e-9}GB with shape {Vh.shape}, dtype {Vh.dtype}')
    #del U #H and U are both of size 
    del H
    gc.collect()
    print('deleted H, U')
    print(f'Labels have shape {L.shape}')
    Who =  np.dot(np.dot(L.T, uh) / s[:n], v) 
    #alpha = s[0]*0.001
    #print(f'Tikhonov with parameter {alpha}, largest Singualar value: {s[0]:.3e}')
    #Who =  np.dot(s*np.dot(L, Vh.T) / (alpha +s**2), U.T)

    return Who

def lstsq_pcr(H, labels, thresh=1e-4):
    global Vh

    if labels.ndim != 2:
        raise ValueError("Labels must have shape (time, features)")
    print('Doing svd')
    U, s, Vh = scipy.linalg.svd(H, full_matrices=False)
    scale = s[0]
    n = np.sum(np.abs(s/scale) > thresh)  # Ensure condition number less than 1/thresh
    
   
    #Reduced data matrix
    H_r = H.dot(Vh.T)
    print(f'H has shape {H.shape}')
    print(f'H_r has shape {H_r.shape}')
    
    
    U, s, Vh = scipy.linalg.svd(H_r.T, full_matrices=False)
    scale = s[0]
    n = np.sum(np.abs(s/scale) > thresh)  # Ensure condition number less than 1/thresh
    
    L = labels.T
    v = Vh[:n, :].T
    uh = U[:, :n].T  

    
    print('deleted H, U')
    #Who =  np.dot(np.dot(L, v) / s[:n], uh) 
    Who =  np.dot(np.dot(L, v) / s[:n], uh) 
   
    return Who




def imed_lstsq_stable(states, inputs, labels, sigma):
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    flat_labels = labels.reshape(inputs.shape[0], -1)

    # prep IMED
    G = imed_matrix(inputs.shape[1:], sigma=sigma)
    (w,V) = np.linalg.eigh(G)

    #thresh = 0
    #n = np.sum(np.abs(w) < thresh)
    #s = np.sqrt(w[n:])
    #V = V[n:,:]
    s = np.sqrt(w)

    G12 = np.dot(V.T, s[:,None]*V)

    # transform inputs / labels
    flat_inputs = np.matmul(G12, flat_inputs[:,:,None])[:,:,0]
    flat_labels = np.matmul(G12, flat_labels[:,:,None])[:,:,0]

    # compute Who
    Who  = lstsq_stable(states, flat_labels)
    s    = 1/s
    iG12 = np.dot(V.T,s[:,None]*V)
    Who  = np.dot(iG12, Who)
    return Who
