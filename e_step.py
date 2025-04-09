import numpy as np
import quantities as pq
from sklearn.decomposition import FactorAnalysis
import scipy.linalg as linalg
from elephant.gpfa.gpfa_util import inv_persymm, logdet


def make_K_big(params, T):
    # this function makes the \bar{K} as in equation A3
    """
    % INPUTS:
    %
    % params       - GPFA model parameters
    % T            - number of timesteps
    %
    % OUTPUTS:
    %
    % K_big        - GP covariance matrix with dimensions (xDim * T) x (xDim * T).
    %                The (t1, t2) block is diagonal, has dimensions xDim x xDim, and 
    %                represents the covariance between the state vectors at
    %                timesteps t1 and t2.  K_big is sparse and striped.
    % K_big_inv    - inverse of K_big
    % logdet_K_big - log determinant of K_big
    """
    ### for each row's GP, we will need to spread out its entries to various blocks of the bigK.
    ### if T = 3, GP = 2
    ### --GP1-- abc, where a = K_1(0,0),  b = K_1(0,1),  c = K_1(0,2)
    ### --GP2-- def
    ###
    ### will turn into
    ###
    ### a 0 | b 0 | c 0
    ### 0 d | 0 e | 0 f
    ### ---   ---   ---
    ### b 0 |need | need
    ### 0 e |K(1,1)| K(1,2)
    ### ---   ---   ---
    ### c 0 |need | need
    ### 0 f |K(1,2)|K(2,2)
    xDim = np.shape(params["C"])[1]
    
    # initialize matrix
    K_big = np.zeros((xDim * T, xDim * T))
    K_big_inv = np.zeros((xDim * T, xDim * T))
    logdet_K_big = 0
    Tdif = np.tile(np.arange(T), (T, 1)).T - np.tile(np.arange(T), (T, 1))
    
    # now, for each factor, spread out
    for i in range(xDim):
        # make the small K first
        K = (1 - params['eps'][i]) * np.exp(-params['gamma'][i] / 2 * Tdif ** 2) \
            + params['eps'][i] * np.eye(T)
        
        # spread out
        K_big[i::xDim, i::xDim] = K
    
        # inverses
        K_big_inv[i::xDim, i::xDim] = np.linalg.inv(K)
        logdet_K = logdet(K)
        logdet_K_big = logdet_K_big + logdet_K
    
    
    return K_big, K_big_inv, logdet_K_big
    

def e_step(seqs, params):
    ### E-step
    ### Followed pretty much as the original GPFA implementation by Bryon Yu and John Cunningham.
    ### Also referred the elephant implementation. [https://github.com/NeuralEnsemble/elephant/blob/master/elephant/gpfa/gpfa_core.py]
    ### Some simplification at the expense of efficiency
    ### Included more annotations
    ### Shijie Gu, shijiegu@berkeley.edu
    ###
    ### Input:
    ### seqs: numpy recarray, 
    ###       y (yDim x T) -- neural data
    ###       T (1 x 1)    -- number of timesteps
    ### params, GPFA parameters: R, C, d
    ### Output:
    ### seq with added fields:
    ###       xsm (xDim x T) -- posterior mean at each time point 
    ###.      Vsm (xDim x xDim x T) -- posterior covariance at each timepoint
    ###       VsmGP (T x T x xDim)  -- posterior covariance of each GP
    ### LL - data log likelihood log(p(Y = y))
    
    # copy the contents of the input data structure to output structure
    T = seqs["T"][0]
    dtype_out = [(x, seqs[x].dtype) for x in seqs.dtype.names]
    dtype_out.extend([('xsm', object), ('Vsm', object),
                      ('VsmGP', object)])
    seqs_out = np.empty(len(seqs), dtype=dtype_out)
    for dtype_name in seqs.dtype.names:
        seqs_out[dtype_name] = seqs[dtype_name]

    ## Precomputations
    yDim, xDim = params["C"].shape
    Rinv = np.diag(1./np.diag(params["R"])) #diag
    logdet_R = np.sum(np.log(np.diag(params["R"])))

    CRinv = (1./np.diag(params["R"]).reshape((1,-1))) * params["C"].T # use broadcasting, it is C.T @ Rinv
    CRinvC = CRinv @ params["C"] #yDim x xDim
    blah = [CRinvC for _ in range(T)]
    CRinvC_big = linalg.block_diag(*blah)  # (xDim*T) x (xDim*T)

    K_big, K_big_inv, logdet_K_big = make_K_big(params, T)
    invM, logdet_M = inv_persymm(K_big_inv + CRinvC_big, xDim)

    ## Note that posterior covariance does not depend on observations, 
    ## so can compute once for all trials with same T.
    ## xDim x xDim posterior covariance for each timepoint
    ## Vsm is the cov in <X:j, X:j'> = cov + mean*mean^T
    Vsm = np.zeros((xDim, xDim, T)) + np.nan
    for t in range(T):
        # just pick the corresponding rows and columns to that time
        # it is just invM because of the Woodbury matrix identity.
        # See text for derivation
        Vsm[:,:,t] = invM[np.arange(t,t+xDim),:][:,np.arange(t,t+xDim)]

    VsmGP = np.zeros((T, T, xDim)) + np.nan
    for i in range(xDim):
        # the indices are just like how we made the K big.
        # see make_K_big() for details
        VsmGP[:,:,i] = invM[i::xDim,:][:, i::xDim]


    # dif is yDim x (T * trial number)
    dif = np.hstack(seqs['y']) - params['d'][:, np.newaxis] # use broadcasting for each trial

    # term1Mat is (xDim*T) x length(nList)
    term1_mat = (CRinv @ dif).reshape((xDim * T, -1), order='F')
    #         (xDim x yDim) * (yDim x sum(T))

    # Compute CRinvC_big * invM
    xsmMat = K_big @ (np.eye(xDim * T) - CRinvC_big @ invM) @ term1_mat


    for n in range(len(seqs)):
        seqs_out[n]['xsm'] = xsmMat[:, n].reshape((xDim, T), order='F')
        seqs_out[n]['Vsm'] = Vsm
        seqs_out[n]['VsmGP'] = VsmGP

    # Compute data likelihood
    val = -T * logdet_R - logdet_K_big - logdet_M \
            - yDim * T * np.log(2 * np.pi)

    ll = len(seqs_out) * val - (Rinv @ dif * dif).sum() \
                    + (term1_mat.T @invM * term1_mat.T).sum()
    # val term is the log(det(CKC' + R)) term
    # ll term is the log((Y-d)^T(R^-1 - R^-1CinvMR^-1)(Y-d)) term

    return seqs_out, ll


