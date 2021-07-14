import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

#from esn_dev.jaxsparse import sp_dot
from sklearn.decomposition import KernelPCA, PCA

def initialize_dynsys(map_ih, hidden_size, spectral_radius=1.5, neuron_connections=10, neuron_dist='uniform',dtype=None):
    """
    Create an ESN dynamical system with input/hidden weights represented as a tuple:

        esn = (Wih, Whh, bh)

    The hidden-to-hidden matrix (the reservoir) is a sparse matrix in turn represented
    as a tuple of values, row/column indices, and its dense shape:
        Whh = (((values, rows, cols), shape)
   
   Arguments:
        map_ih: An `esn_dev.input_map.InputMap`
        hidden_size: ESN hidden size
        spectral_radius: spectral radius of Whh
        neuron_connections: number of non-zero values in rows of Whh
        neuron_dist: distribution of non-zero values in Whh (uniform/normal)
    
    Returns:
        (Wih, Whh, bh)
    """
    nonzeros_per_row = int(neuron_connections)
    
    Whh = reservoir(
        hidden_size, 
        spectral_radius, 
        nonzeros_per_row,
        neuron_dist,
        dtype)
    
    # csr matrix more efficient for matrix product
    Whh = Whh.tocsr()
    
    bh  = np.asarray(
        np.random.uniform(-1, 1, hidden_size),
        dtype=dtype)

    return (map_ih, Whh, bh)

def evolve_hidden_state(dynsys, xs, h,mode=None):
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
    dtype=xs.dtype
    (map_ih, Whh, bh) = dynsys[:3]
    
    if mode == 'predict':
        # returns only last hidden state
        h = fwd_prop(h=h,x=xs,Whh=Whh,Win=map_ih)
        return h

    # number of time steps
    if xs.ndim == 2:
        #assume one 2D-sample
        if mode!='predict':
            print("Single sample received. Using mode 'predict'")
            mode = 'predict'
            evolve_hidden_state(dynsys, xs, h, mode)
            
    elif xs.ndim == 3:
        T = xs.shape[0]
    
    if mode=='transient':
        # returns only last hidden state
        for t in range(T):
            h = fwd_prop(h=h,x=xs[t],Whh=Whh,Win=map_ih)
        return h

    elif mode == 'train':
            
        # Dimension of hidden state
        Nhidden = Whh.shape[0]

        H = np.zeros([T,Nhidden],dtype=dtype)
        H[0,:] =  fwd_prop(h=h,x=xs[0],Whh=Whh,Win=map_ih)

        
        for t in range(1,T):
            H[t,:] = fwd_prop(h=H[t-1,:],x=xs[t],Whh=Whh,Win=map_ih)
        
        return H
        
    else:
        print("Evolution mode should be 'transient', 'train' or 'predict'" )
        return
    
        
def fwd_prop(h,x,Whh,Win):
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
    
    return np.tanh(Whh.dot(h) + Win(x))


def reservoir(dim, spectral_radius, nonzeros_per_row,neuron_dist,dtype=None, upper_sr_calc_dim=5000):
    """
    Create a sparse reservoir, Whh, with dimension `dim` x `dim`
    with the following properties:
    - Spectral radius is approximately `spectral_radius`
    - Each row has `nonzeros_per_row` elements (neuron connections)
    - nonzero elements are distributed according to `neuron_dist`
    
    This methods exploits heuristic circular law to design spectral radius.
    As the law is approximate, and grows more precise when `dim` is increased
    the spectral radius is explicitly calculated with ARPACK wrapper
    scipy.sparse.linalg.eigs() when `dim` is less than `upper_sr_calc_dim`.
      
    Params:
        dim:               hidden dimension. Returned matrix Whh
                           is `dim` x `dim`.
        spectral_radius:   magnitude of largest eigenvalue of Whh
        nonzeros_per_row:  fixed number of nonzero elements in rows of Whh
        neuron_dist:       distribution of values of nonzero elements
                           `uniform` or `normal`. Location of elements
                           always uniformly spread.
        dtype:             dtype of Whh.
        upper_sr_calc_dim: dimension below which to explicitly determine
                           spectral radius instead of circular law approach
    Returns: 
        Whh:               sparse coo_matrix of shape `dim` x `dim`
    """
    print(f'Using {nonzeros_per_row} nonzeros per row and {neuron_dist} distribution')
    
    #Use current random state as seed
    seed = np.random.get_state()[1][0]
    rng = np.random.default_rng(seed)
    
    # number of values in sparse matrix
    nr_values = dim * nonzeros_per_row
    
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
               (nonzeros_per_row*(1+k*dim**(-k))**2))
        
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
               (nonzeros_per_row*(1+k*dim**(-k))**2))
        
        dist_args = dict(
            loc=0.0,
            scale=np.sqrt(var),
            size=[nr_values])
    
    
    # Generate sparse matrix in (vals, (row_idx, col_idx)) format
    
    dense_shape = (dim, dim)

    # get row_idx like: [0,0,0,1,1,1,....]
    row_idx = np.tile(np.arange(dim)[:, None], nonzeros_per_row).reshape(-1)

    # get col idx that are unique within each row
    col_idx = []
    for ii in range(dim):
        #random generate without replacement
        cols = tuple(rng.choice(dim, size=nonzeros_per_row, replace=False))
        col_idx += (cols)
    col_idx = np.asarray(col_idx)
    vals = dist_gen(**dist_args,)
    
    # build sparse coo matrix (efficient for building)
    matrix = coo_matrix((vals, (row_idx, col_idx)),shape=dense_shape,dtype=dtype)

    # matrix now has approximate spectral radius
    # if low-dim, do manual calc:
    if dim <= upper_sr_calc_dim:

        eig_max = eigs(
            matrix, 
            k=1, 
            tol=0,
            return_eigenvectors=False,
            which='LM',
            ncv=200,
            v0 = np.ones(dim)
            )
        
        rho = np.abs(eig_max)
        matrix = matrix.multiply(spectral_radius / rho) 
    
    return matrix

def dimension_reduce(h,pca_object=None,PCs = None):
    """
    Function to fit-transform or transform if already fitted
    
    Params:
        h: single hidden state or state matrix (H)
        pca_object: already-fitted PCA transform or None
                    if not yet fitted
    Returns: 
        if pca_object is None:
            H_r: dimension-reduced hidden state matrix
            pca_object: trained transform object
        else:
            h_r: dimension-reduced hidden state
    """
    
    if pca_object is None:
        
        # Fit the pca_object. 
        # PCs must be specified (int)
        H = h
        pca_object = PCA(n_components=PCs+1)
        H_r = pca_object.fit_transform(H)
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
            h_r = h_r.reshape(-1)
            
            #keep a bias term
            h_r[-1] = 1.
            
        elif h.ndim == 2:
            
            # transform
            h_r = pca_object.transform(h)
            
            #several states
            h_r[:,-1] = 1. 
            
        return h_r