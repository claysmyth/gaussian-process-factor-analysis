# Post-processing
import numpy as np
import scipy as sp
from elephant.gpfa.gpfa_util import segment_by_trial

def post_processing(params_est, seq):
    C = params_est['C']
    X = np.hstack(seq['xsm'])
    latent_variable_orth, Corth, _ = orthonormalize(X, C)
    seq_out = segment_by_trial(
        seq, latent_variable_orth, 'latent_variable_orth')

    params_est['Corth'] = Corth
    return params_est, seq_out

def orthonormalize(x, C):
    """
    As in elephant package, annotated by Shijie
    Orthonormalize the columns of the loading matrix and apply the
    corresponding linear transform to the latent variables.

    Parameters
    ----------
    x :  (xDim, T) np.ndarray
        Latent variables
    C :  (yDim, xDim) np.ndarray
        Loading matrix

    Returns
    -------
    latent_variable_orth : (xDim, T) np.ndarray
        Orthonormalized latent variables
    Lorth : (yDim, xDim) np.ndarray
        Orthonormalized loading matrix
    TT :  (xDim, xDim) np.ndarray
       Linear transform applied to latent variables
    """
    
    xDim = C.shape[1]
    U, D, V = sp.linalg.svd(C)
    
    ####
    # Since we have C x
    #.            UDV x = U(DV x), we want the (DV x)
    ####
    # TT is transform matrix
    TT = np.diag(D) @ V

    Lorth = U
    latent_variable_orth = TT @ x
    
    return latent_variable_orth, Lorth, TT
