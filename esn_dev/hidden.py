import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

#from esn_dev.jaxsparse import sp_dot
from sklearn.decomposition import PCA as PCA

def initialize_dynsys(map_ih, hidden_dim,
                      spectral_radius=1.5, neuron_connections=10, neuron_dist='uniform',dtype=None,upper_sr_calc_dim=5000, **kwargs):
    """
    Create an ESN dynamical system with input/hidden weights represented as a tuple:

        esn = (Wih, Whh, bh)

    The hidden-to-hidden matrix (the reservoir) is a sparse matrix in turn represented
    as a tuple of values, row/column indices, and its dense shape:
        Whh = (((values, rows, cols), shape)
   
   Arguments:
        map_ih: An `esn_dev.input_map.InputMap`
        hidden_dim: ESN hidden state size
        spectral_radius: spectral radius of Whh
        neuron_connections: number of non-zero values in rows of Whh
        neuron_dist: distribution of non-zero values in Whh (uniform/normal)
    
    Returns:
        (Wih, Whh, bh):  Initialization of dynamical system.
    """
    
    Whh = reservoir(
        hidden_dim, 
        spectral_radius, 
        neuron_connections,
        neuron_dist,
        upper_sr_calc_dim,
        **kwargs
    )
    
    # csr matrix more efficient for matrix product
    Whh = Whh.tocsr()
    
    """bh  = np.asarray(
        np.random.uniform(-1, 1, hidden_dim),
        dtype=dtype)"""
    
    bh = np.zeros(hidden_dim)

    return (map_ih, Whh, bh)

def evolve_hidden_state(dynsys, xs, h, mode=None,dtype=None):
    """
    Echo State Harvestor.
    Apply and ESN defined by dynsys (as in created from `initialize_dynsys`) to
    each input in xs with the initial state h. Each new input uses the updated
    state from the previous step.
    Arguments:
        dynsys: An ESN tuple (Wih, Whh, bh)
        xs: Array of inputs. Time in first dimension, unless single sample
        h: Initial hidden state
        mode: evolution setting: 'transient', 'train' or 'predict' 
    Returns:
        h: Final hidden state (for modes 'predict' or 'transient')
        or
        H: All hidden states (for mode 'train')
    """
    (map_ih, Whh, bh) = dynsys[:3]
    
    if mode == 'predict':
        # returns only last hidden state
        h = fwd_prop(h=h,x=xs,Whh=Whh,Win=map_ih,bh=bh)
        return h

    # number of time steps
    if xs.ndim == 2:
        #assume one 2D-sample
        if mode!='predict':
            print("Single sample received. Using mode 'predict'")
            mode = 'predict'
            h = evolve_hidden_state(dynsys, xs, h, mode)
            return h
            
    elif xs.ndim == 3:
        T = xs.shape[0]
    
    if mode=='transient':
        # returns only last hidden state
        for t in range(T):
            h = fwd_prop(h=h,x=xs[t],Whh=Whh,Win=map_ih,bh=bh)
        return h

    elif mode == 'train':
            
        # Dimension of hidden state
        Nhidden = Whh.shape[0]

        H = np.zeros([T,Nhidden],dtype=dtype)
        H[0,:] =  fwd_prop(h=h,x=xs[0],Whh=Whh,Win=map_ih,bh=bh)

        
        for t in range(1,T):
            H[t,:] = fwd_prop(h=H[t-1,:],x=xs[t],Whh=Whh,Win=map_ih,bh=bh)
        
        return H
        
    else:
        print("Evolution mode should be 'transient', 'train' or 'predict'" )
        return
    
        
def fwd_prop(h,x,Whh,Win,bh):
    """
    Recurrent Neural Network Forward Propagation
    Arguments:
        h:       Initial hidden state
        x:       Driving input
        Whh:     Square (sparse) hidden-to-hidden matrix
        Win: Function like `esn_dev.input_map.InputMap` 
                 to transform x to Nhidden = Whh.shape[0]

    Returns:
        h:       Next hidden state (t --> t+1)
     """
    warn_limit = 10
    Win_x = Win(x)
    if np.abs(Win_x).max() > 10:
        print('Warning: Input map function Win(x) gave'
              'values with magnitude larger than 10.'
              'tanh() activation has range (-1;1), so if all values'
             'are very high, they will be squashed to the same output')
    
    return np.tanh(Whh.dot(h) + Win_x + bh)


def reservoir(hidden_dim, spectral_radius, neuron_connections,neuron_dist='uniform', upper_sr_calc_dim=5000,dtype=None,**kwargs):
    """
    Create a sparse reservoir, Whh, with dimension `hidden_dim` x `hidden_dim`
    with the following properties:
    - Spectral radius is approximately `spectral_radius`
    - Each row has `neuron_connections` elements (neuron connections)
    - nonzero elements are distributed according to `neuron_dist`
    
    This methods exploits heuristic circular law to design spectral radius.
    As the law is approximate, and grows more precise when `hidden_dim` is increased
    the spectral radius is explicitly calculated with ARPACK wrapper
    scipy.sparse.linalg.eigs() when `hidden_dim` is less than `upper_sr_calc_dim`.
      
    Params:
        dim:               hidden dimension. Returned matrix Whh
                           is `hidden_dim` x `hidden_dim`.
        spectral_radius:   magnitude of largest eigenvalue of Whh
        neuron_connections:fixed number of nonzero elements in rows of Whh
        neuron_dist:       distribution of values of nonzero elements
                           `uniform` or `normal`. Location of elements
                           always uniformly spread.
        dtype:             dtype of Whh.
        upper_sr_calc_dim: dimension below which to explicitly determine
                           spectral radius instead of circular law approach
    Returns: 
        Whh:               sparse coo_matrix of shape `hidden_dim` x `hidden_dim`
    """
    print(f'Using {neuron_connections} nonzeros per row and {neuron_dist} distribution')
    
    #Use current random state as seed
    seed = np.random.get_state()[1][0]
    rng = np.random.default_rng(seed)
    
    # number of values in sparse matrix
    nr_values = hidden_dim * neuron_connections
    
    if not (neuron_dist=='uniform' or neuron_dist=='normal'):
        print(f"neuron_dist {neuron_dist} unknown.\n"
              "Should be 'uniform' or 'normal'\n"
              "Proceeding with 'uniform'")
        neuron_dist = 'uniform'
    
    if neuron_dist == 'uniform':
        dist_gen = np.random.uniform

        #heuristic distribution param
        k = 0.4098 #pm 0.0003
       
        # Pick variance according to 
        # desired spectral radius        
        var = (spectral_radius**2   /
               (neuron_connections*(1+k*hidden_dim**(-k))**2))
        
        #pick 0-symmetric interval
        high = np.sqrt(3*var)
        low  = -high
        
        dist_args = dict(
            high=high,
            low=low,
            size=[nr_values],
        )
        
    elif neuron_dist == 'normal':
        dist_gen = np.random.normal

        #heuristic distribution param
        k = 0.4075 #pm 0.0003
        
        # Pick variance according to 
        # desired spectral radius
        var = (spectral_radius**2   /
               (neuron_connections*(1+k*hidden_dim**(-k))**2))
        
        dist_args = dict(
            loc=0.0,
            scale=np.sqrt(var),
            size=[nr_values])
    
    
    # Generate sparse matrix in (vals, (row_idx, col_idx)) format
    
    dense_shape = (hidden_dim, hidden_dim)

    # get row_idx like: [0,0,0,1,1,1,....]
    row_idx = np.tile(np.arange(hidden_dim)[:, None], neuron_connections).reshape(-1)

    # get col idx that are unique within each row
    col_idx = []
    for ii in range(hidden_dim):
        #random generate without replacement
        cols = np.asarray(
            rng.choice(
                
                hidden_dim, 
                size=neuron_connections, 
                replace=False),
            
            dtype=dtype
        )
        col_idx += tuple(cols)
    col_idx = np.asarray(col_idx)
    vals = dist_gen(**dist_args,)
    
    # build sparse coo matrix (efficient for building)
    matrix = coo_matrix((vals, (row_idx, col_idx)),shape=dense_shape,dtype=dtype)

    # matrix now has approximate spectral radius
    # if low-dim, do manual calc:
    if hidden_dim <= upper_sr_calc_dim:

        eig_max = eigs(
            matrix, 
            k=1, 
            tol=0,
            return_eigenvectors=False,
            which='LM',
            ncv=200,
            v0 = np.ones(hidden_dim)
            )
        
        rho = np.abs(eig_max)
        matrix = matrix.multiply(spectral_radius / rho) 
    
    return matrix

def dimension_reduce(h,pca_object=None,n_PCs = None):
    """
    Function to fit-transform or transform if already fitted.
    if n_PCs is none (relevant when teaching PCA), we disable
    PCA.
    
    Params:
        h: single hidden state (h) or state matrix (H)
           shaped (T x Nhidden)
        pca_object: already-fitted PCA transform or None
                    if not yet fitted
    Returns: 
        if pca_object is None:
            H_r: dimension-reduced hidden state matrix
                 shape (T x n_PCs)
            pca_object: trained transform object
        else:
            h_r: dimension-reduced hidden state
    """
    if n_PCs is None and pca_object is None:
        if h.shape[0]>1:
            print('no pca')
            return h, None
        else:
            return h
    
    elif pca_object is None:
        # PCA not yet fitted.
        # n_PCs is not None
        # in this case

        # Fit the pca_object.         
        H = h
        print(f'Using {n_PCs} principal components, and min(H.shape)={min(H.shape)}')
        if n_PCs > min(H.shape):
            n_PCs = min(H.shape)-1
            print('n_PCs can be maximally min(Hm,Hn), using n_PCs = {n_PCs}')
        pca_object = PCA(n_components=n_PCs,whiten=False)
        H_r = pca_object.fit_transform(H)
        
        """# feature mean should be subtracted
        # when using PCA, for components
        # to have variance interpretation
        mean = H.mean(axis=0)
        H -= mean[np.newaxis,:]
        
        from scipy.linalg import svd
        _,_,V = svd(H, full_matrices=False)

        #V = V[:n_PCs+1,:].T  #+1 for bias term
        #V = (V.T)[:,:n_PCs+1]
        V = V[:n_PCs+1,:]
        #transform H
        H_r = H.dot(V.T)
        class pca(object):
            def __init__(self,V,mean):
                self.V  = V
                self.mean = mean
            def transform(self,h):
                return V.dot(h-self.mean)#().dot(self.V)
        
        pca_object=pca(V,mean)
        """
        #keep a bias term        
        H_r[:,-1] = 1. 
        
        return H_r, pca_object
    
    else:
        
        # transform the hidden state(s)
        # using already trained pca_object
        
        if h.ndim == 1:
            # reshape to signal single sample
            h = h.reshape(1, -1)
            #transform
            h_r = pca_object.transform(h)
            #single state
            h_r = np.squeeze(h_r)#h_r.reshape(-1)
            #keep a bias term
            h_r[-1] = 1.
            
        elif h.ndim == 2:
            # transform
            h_r = pca_object.transform(h)
            #several states
            h_r[:,-1] = 1. 
            
        return h_r