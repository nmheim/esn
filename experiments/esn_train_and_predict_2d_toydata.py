import numpy as np
from esn_dev.input_map import InputMap
from esn_dev.toydata import gauss2d_sequence, mackey2d_sequence
from esn_dev import hidden, optimize, readout
from esn_dev.utils import *
from esn_dev.visualize_results import MSE_over_time, animate_comparison
from time import time
import gc

def train_and_predict_2d(data, specs,
                             savedir='tmp',
                             spectral_radius=1.5,
                             neuron_connections=10,
                             Ntrans=500,        # Number of transient initial steps before training
                             Ntrain=2500,       # Number of steps to train on
                             Npred=500,         # Number of steps for free-running prediction
                             plot_prediction=False,
                             sigma=(1,5,5),
                             eps=1e-5,
                             n_PCs=None,
                             dtype=np.float64,
                             lstsq_method ='scipy',
                             lstsq_thresh = None,
                             ST_method = 'DCT',
                             cpus_to_use = 32,
                             scale_min = -1,
                             scale_max =  1,
                             neuron_dist = 'normal',
                             upper_sr_calc_dim=5000,
                             save_condition = 'always',   
                             random_seed = None,
                             config = None,
                             **kwargs):
    
    """
    Given data and configurations, use the scalable spatial echo state network
    to train on video-like input and predict future video frames in
    free-running-mode (predictions normally diverge from targets over time).
    
    ---
    What is the scalable spatial echo state network?

    ESN consists of dynamical system and a separate readout/prediction layer:
    The dynamical system is recurrent, and input driven, i.e.:
    
                        h[t+1] = tanh( Win . x+Whh . h[t] + bh ),
                        
    where h is a high-dimensional hidden state with some memory, x is the input,
    Win is a matrix mapping x-->h, Whh is the 'reservoir', a matrix mapping from
    h-->h. bh is an optional bias vector with dimension as h. Image input is flatened
    to vector format x.
    
    The prediction layer then uses h from training steps as regressors,
    gathered in a matrix H:
    
                                y[t+1] = Who.h[t]
                                
    Where Who is an output matrix mapping from h-->y (output, same dimesion
    as input, x) that is optimized for this purpose using the efficient 
    multiple least squares method; only one time. No gradient-based learning here. 
    
    Win and Whh are not optimized, and must be selected for the ESN to give good
    predictions. A number of parameters control this, like 'specs' (for Win) and 
    'spectral radius' (for Whh).
    --- 
    
    if 'save_condition' is True or 'always', save model and prediction
    arrays as '.npy' files along with the configuration dictionary 'config'
    (saved in human-readable .yaml, and machine-readable .pkl formats)
    if plot_prediction is True, also save an mp4-video comparing predictions
    with a target data set, unseen by the ESN, and a plot of MSE over time. 
    
    Params:
        data:        (T,M,N)-ndarray, where T is time, M,N are spatial.
        specs:       [list] Input map specifications, that create Win matrix.
                     Example:
        [{'type':'pixels','size':input_shape,'factor':1.0},
        {'type' :'conv','size':(9, 9),'kernel':'gauss','factor': 1.0},
        {'type' :'gradient','factor': 1.0},
        {'type' :'random_weights','input_size':(input_size),'hidden_size':1000,'factor':1.},
        {"type" :'gradient', 'factor' 1.},
        {"type" :'dct', 'size':(15,15), 'factor': 1}]

        savedir:            project folder to save at least one esn model in
                            if 'save_condition' is also fulfilled. The model is
                            saved in 'savedir/esn001', and if run again
                            'savedir/esn002', with an mse-overview in 'savedir'.
        spectral_radius:    spectral radius of reservoir Whh. Values around 1
                            are a good start. Crucial for network stability,
                            and good predictions.
        neuron_connections: Number of non-zero elements in each
                            row of reservoir matrix Whh.
        Ntrans:             Duration of part of training set _not_ to regress
                            on, but to use for warming up the dynamical system,
                            giving better hidden states, h. trans: Transient.
        Ntrain:             Total duration of training set (including Ntrans).
        Npred:              Number of prediction steps to use. Data set must be
                            large enough to have N=Ntrain+Npred steps so that
                            predictions can be compared to unseen target obs.
        plt_prediction:     (bool) Whether to plot animation and error-plots.
                            Only used if 'save_condition' is fulfilled.
        ST_method:          'FFT' or 'DCT'. Used for preprocessing of video to
                            enhance learning. ST is standardizing transform
                            inherent to the IMED: IMage Euclidean Distance.
                            FFT and DCT are frequency methods.
        sigma:              Parameter of spatial IMED pre-processing of the data
                            to enhance learning. Larger sigma blurs out image
                            more. If scalar, sigma is applied along both time
                            and spatial axes. Otherwise, must be specified as
                            sigma=(1,5,5) for 1 along time axis and 5 along 
                            both spatial axes (normally a better option.)
        eps:                Parameter of IMED-proprocessing of the data. 
                            The larger eps is, the more noise is suppressed
                            to allow post-processing to work (operations are
                            opposite of pre-processing steps). Should be a
                            small positive number, growing with 'sigma'.
                            1e-2 is usually a good choice.
        cpus_to_use:        Number of cpu cores to use for parallel standardizing
                            transform preprocessing and postprocessing.
                            Must be an int. If -1, uses all cores. Equal to
                            scipy.fft.___ 'workers' keyword.
        n_PCs:              Number of trained parameters to fit per pixel
                            in the image (video slice). Equally, it is 
                            the number of principal components used in a
                            PCA dimension reduction to improve scalability
                            and training performance.
        dtype:              Data type of the ESN-structures (not necessarily)
                            equal to data-dtype. np.float64, or 'float64' recommended,
                            as low precision propagates due to iterative modelling.
        lstsq_method:       If passed, choice of least squares method.
        lstsq_thresh:       Parameter of the 'lstsq_method' of choice. Allows fitting
                            with rank-deficient matrices, adds robustness.
        scale_min:          Scaled minimum of the training set. -1 is recommended,
                            as it is the minimum of the activation function tanh().
        scale_min:          Scaled maximum of the training set. 1 is recommended,
                            the maximum of activation function tanh()
        neuron_dist:        'uniform' or 'normal'. Distribution of random non-zero
                            values of matrix reservoir Whh.
        upper_sr_calc_dim:  (int) Used to adapt Whh to the chosen 'spectral_radius'.
                            If increased above dimension of h, inefficient but more precise
                            calculations are made. Dimension of h is a result of 'specs'
        save_condition:     Whether to save model output and plot (if 'plot_predictions').
                            Can be True, or False, to enable or disable,
                            or 'always', and 'never' for same functionality.
                            or 'if_better', to only save if lower mse com any previous 
                            models was achieved.
        random_seed:        Seed to inialize (and recover) random generator used for e.g.
                            creation of reservoir matrix Whh. Can be int or None.
        config:             Dictionary of all loaded parameters that is also used as 
                            input to this function using (...,config=config,**config).
                            If not passed, parameters and updated values from run
                            cannot be saved (when 'save_condition' is fulfilled).
    Returns:
        mse:                MSE of predictions versus unseen prediction targets.
    """
    
    #start timing of entire function, except plotting and output IO
    start_tottime = time()
    
    #Save initial seed
    np.random.seed(random_seed)
    seed = str(np.random.get_state()[1][0])
    
    # update config to save config at end
    config['random_seed'] = seed

    N = Ntrain + Npred + 1
    assert data.ndim == 3
    assert data.shape[0] >= N
    # Shape of video frames
    img_shape = data.shape[1:]

    
    # prepare data
    train_inputs, train_targets, pred_targets = split_train_label_pred(data,Ntrain,Npred)
    print(f'Data is taking up {data.nbytes*1e-9:.1f}GB. Deleting after splitting.')
    del data
    gc.collect()
    
    # make copy to not affect original prediction targets
    pred_targets_ST = np.copy(pred_targets)
        
    # Pre-processing of data set, along with training_min
    # and training_max to process from train_input
    # is used throughout.
    train_inputs, training_min, training_max = preprocess(
        data=train_inputs,
        training_min = None,
        training_max = None,
        **kwargs
    )
    # must update configuration 
    # to store settings at end
    config['training_min']=training_min
    config['training_max']=training_max
    
    # Do standardizing transform preprocessing
    # followed by scaling between training_min and training_max
    train_targets = preprocess(data=train_targets,**config)
    pred_targets_ST = preprocess(data=pred_targets,**config)
    
    # time all parts of ESN
    timings = {}
    start_init = time()
    
    # Create inputMap function using 'specs' that act as Win for Win@x (see docstring)
    map_ih = InputMap(specs)
    
    #  hidden dimension, according to input 'specs'
    hidden_size = map_ih.output_size(img_shape)
    
    print(f"Input size (1 slice): {train_inputs[0].shape}\nHidden size: {hidden_size}")
    esn = hidden.initialize_dynsys(map_ih, hidden_size,spectral_radius,neuron_connections,dtype=dtype)
    end_init = time()
    print(f'Building Whh took {end_init-start_init:.2f}s')
        

    start_trans = time()
    # compute final state after transient warmup of hidden state
    # Use only Ntrans-part of training set, no optimization on this.
    h_trans = hidden.evolve_hidden_state(esn, train_inputs[:Ntrans], h=np.zeros(hidden_size),mode='transient')
    end_trans   = time()
    print(f'Transient evolution of dynamical system ({Ntrans} steps) took {end_trans-start_trans:.2f}s')
   
    
    # Use only last state after transient phase as start
    # Now harvest all states of remaining training set here, in H.
    start_harvest = time()
    H = hidden.evolve_hidden_state(esn, train_inputs[Ntrans:], h=h_trans,mode='train')
    # delete trainin_input not used anymore
    del train_inputs
    
    gc.collect()
    # Initial input and hidden state for predictions
    y0, h0 = train_targets[-1], H[-1]

    end_harvest = time()
    print(f'State harvesting of {Ntrain-Ntrans} echo states took {end_harvest-start_harvest:.2f}s')
   
    # reshape training targets to vectors. Keep time dimension.
    train_targets = train_targets.reshape(train_targets.shape[0], -1)
        
    start_pca = time()
    #reduce hidden state matrix H before using for training
    if n_PCs is not None:
        H, pca_object = hidden.dimension_reduce(H,pca_object=None, n_PCs = n_PCs)
    end_pca = time()
    print(f'PCA Dimension reduction of H took: {end_pca-start_pca:.2f}s')
    

    
    start_lstsq = time()
    esn = optimize.create_readout_matrix(esn, H, train_targets[Ntrans:])
    end_lstsq   = time()
    print(f'Least Squares optimization took {end_lstsq-start_lstsq:.2f}s')
    # Delete targets not used anymore
    del train_targets
    gc.collect()
        
    # predict
    start_pred = time()    
    predictions_ST = readout.predict(esn, y0, h0, Npred=Npred, pca_object=pca_object)  
    end_pred = time()
    print(f'Predicting for {Npred} time steps took {end_pred-start_pred:.2f}s')
    
    # Post processing achieved using inverse=True
    predictions_iST = preprocess(
        data=predictions_ST,
        inverse = True,
        **config
    )
    
        
        
    # Post process the prediction targets
    # not used for unbiased mse_orig!
    pred_targets_iST = preprocess(
        data = pred_targets_ST,
        inverse = True,
        **config
    )
    
    # SCORING
    mse_ST = score(predictions_ST,pred_targets_ST)
    mse_ist = score(predictions_iST,pred_targets_iST)
    mse_orig = score(predictions_iST,pred_targets)    
    mse_list = [
        f'MSE-IMED: MSE in standardizing-transform-space: {mse_ST:.2e}',
        f'MSE after rescaling,iST of preds and targets: {mse_ist:.2e}',
        f'MSE after rescaling,iST of preds wrt. untouched targets: {mse_orig:.2e}'
    ]
    config['Model_Errors'] = mse_list
    for mse_str in mse_list:
        print(mse_str)
    
    end_tottime = time()
    print(f'Total time: {end_tottime-start_tottime:.1f}s')
    
    timings = [
            f'Building of reservoir Whh took {end_init-start_init:.2f}s',
            f'Transient evolution of dynamical system (for {Ntrans} steps) took{end_trans-start_trans:.2f}s',
            f'Harvesting {Ntrain-Ntrans} Hidden Echo States took {end_harvest-start_harvest:.2f}s',
            f'PCA Dimension reduction of H took {end_pca-start_pca:.2f}s',
            f'Least Squares optimization took {end_lstsq-start_lstsq:.2f}s',
            f'Predicting for {Npred} time steps took {end_pred-start_pred:.2f}s',
            f'Total time (no plotting/saving): {end_tottime-start_tottime:.1f}s',
        ]
    
    #add to config dictionary that is output
    config['timings_dictionary'] = timings 

    arrays_to_save = dict(
        y0=y0,
        h0=h0,
        Whh=esn[1],
        bh=esn[2],
        Who=esn[3],
        predictions=predictions_iST,
        targets=pred_targets,
        )
    folder = save(
        targets=pred_targets,
        predictions=predictions_iST,
        dict_of_arrays= arrays_to_save,
        param_dict=config,
        save_condition = config['save_condition'],
        savedir = config['savedir'],
    )
    
    if plot_prediction and folder is not None:
        print(f'Saving plots/animation in {folder}...')
        mse_fig = MSE_over_time(
            targets=pred_targets_ST,
            predictions=predictions_ST,
            subplot_kw = {'ylabel':'IMED-MSE',
                         'xlabel' :'Prediction Time Step'}
        )
        mse_fig.savefig(f'{folder}/MSE_plot.pdf', bbox_inches='tight')

        print('Animating')
        animate_comparison(
            targets=pred_targets,
            predictions=predictions_iST,
            filepath=f'{folder}/comparison.mp4',
            fps=17,
            dpi=150)
            
    return mse_orig

def test_lissajous(savedir):
    # Spatial resolution
    input_shape = (100,100)
    
    # flattened out dimension
    input_size  = input_shape[0] * input_shape[1]
    
    # Specify Training set and Prediction Set Split
    # Note: first 'Ntrans' steps in training set
    # are not regressed upon.
    Ntrain=800
    Npred =150
    Ntrans=100
    
    # total time steps
    N = Ntrain+Npred+Ntrans
    
    #create data
    data = gauss2d_sequence(N=N,size=input_shape,dtype=np.float32)
    
    specs = [
        {"type":"pixels", "size":input_shape, "factor": 2.},
        {"type":"conv", "size":(3,3),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(5,5),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(7,7),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(9,9),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(3,3),   "kernel":"random", "factor": 1.},
        {"type":"conv", "size":(5,5),   "kernel":"random", "factor": 1.},
        {"type":"conv", "size":(7,7),   "kernel":"random", "factor": 1.},
        {"type":"conv", "size":(9,9),   "kernel":"random", "factor": 1.},
        {"type":"gradient", "factor": 1.},
        {"type":"dct", "size":(15,15), "factor": 1},
        {"type":"random_weights", "input_size":input_size, "hidden_size":3500, "factor": 1.},
    ]
    
    parameter_dict = dict(
        specs = specs,
        Npred  = Npred,
        Ntrain = Ntrain,
        Ntrans = Ntrans,
        spectral_radius = 1.,
        neuron_connections = 10,
        n_PCs = 600,
        sigma  =  (0.2,2,2),
        eps    =  1e-3,
        plot_prediction = True,
        dtype='float64',
        lstsq_method ='scipy',
        lstsq_thresh = None,
        ST_method = 'DCT',
        cpus_to_use = 32,
        scale_min = -1,
        scale_max =  1,
        savedir = savedir,
        neuron_dist = 'normal',
        upper_sr_calc_dim=5000,
        save_condition = 'always',
        random_seed = np.random.seed()

    )

    return train_and_predict_2d(data,config=parameter_dict, **parameter_dict)
    

def test_chaotic(savedir):
    # Spatial resolution
    input_shape = (100,100)
    
    # flattened out dimension
    input_size  = input_shape[0] * input_shape[1]
    
    # Specify Training set and Prediction Set Split
    # Note: first 'Ntrans' steps in training set
    # are not regressed upon.
    Ntrain=1200
    Npred =150
    Ntrans=300
    
    # total time steps
    N = Ntrain+Npred+Ntrans
    
    #create data
    data = mackey2d_sequence(N=N,size=input_shape,dtype=np.float32)
    
    specs = [
        {"type":"pixels", "size":input_shape, "factor": 2.},
        {"type":"conv", "size":(3,3),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(5,5),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(7,7),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(9,9),   "kernel":"gauss",  "factor": 1.},
        {"type":"conv", "size":(3,3),   "kernel":"random", "factor": 1.},
        {"type":"conv", "size":(5,5),   "kernel":"random", "factor": 1.},
        {"type":"conv", "size":(7,7),   "kernel":"random", "factor": 1.},
        {"type":"conv", "size":(9,9),   "kernel":"random", "factor": 1.},
        {"type":"gradient", "factor": 1.},
        {"type":"dct", "size":(15,15), "factor": 1},
        {"type":"random_weights", "input_size":input_size, "hidden_size":3500, "factor": 1.},
    ]
    
    parameter_dict = dict(
        specs = specs,
        Npred  = Npred,
        Ntrain = Ntrain,
        Ntrans = Ntrans,
        spectral_radius = 1.,
        neuron_connections = 10,
        n_PCs = 800,
        sigma  =  (0.2,2,2),
        eps    =  1e-3,
        plot_prediction = True,
        dtype='float64',
        lstsq_method ='scipy',
        lstsq_thresh = None,
        ST_method = 'DCT',
        cpus_to_use = 32,
        scale_min = -1,
        scale_max =  1,
        savedir = savedir,
        neuron_dist = 'normal',
        upper_sr_calc_dim=5000,
        save_condition = 'always',
        random_seed = np.random.seed()

    )

    return train_and_predict_2d(data,config=parameter_dict, **parameter_dict)


if __name__ == "__main__":
        mse = test_lissajous("lissajous_blob")
        mse = test_chaotic("chaos_blob")
 