import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import numpy as np
from typing import Dict, List
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Callable
import scipy.io as sio
import time

def m_step_cd(seq: np.recarray, x_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    M-step update for C and d parameters in GPFA. Eq A8.
    """
    # Initialize sum of auto-covariances
    sum_Pauto = np.zeros((x_dim, x_dim))
    
    for trial in range(len(seq)):
        Vsm = seq['Vsm'][trial]
        xsm = seq['xsm'][trial]
        sum_Pauto += (np.sum(Vsm, axis=2) + xsm @ xsm.T)
    
    # Stack observations and states
    Y = np.hstack([seq['y'][trial] for trial in range(len(seq))])  # Stack all y_:t
    Xsm = np.hstack([seq['xsm'][trial] for trial in range(len(seq))])  # Stack all x_:t
    
    # Compute cross terms
    sum_yxtrans = Y @ Xsm.T  # ∑y_:t⟨x_:t⟩'
    sum_xall = np.sum(Xsm, axis=1)  # ∑⟨x_:t⟩
    sum_yall = np.sum(Y, axis=1)  # ∑y_:t
    
    # Construct block matrix for inversion (equation A8)
    term = np.block([
        [sum_Pauto, sum_xall[:, None]],
        [sum_xall[None, :], np.sum([seq["T"][trial] for trial in range(len(seq))])]
    ])
    
    # Compute [C d] using equation A8
    Cd = np.hstack([sum_yxtrans, sum_yall[:, None]]) @ np.linalg.inv(term)
    
    # Split result into C and d
    C = Cd[:, :x_dim]
    d = Cd[:, -1]
    
    return C, d

def m_step_R(seq: np.recarray, C: np.ndarray, d: np.ndarray, force_diagonal: bool = True) -> np.ndarray:
    """
    M-step update for observation noise covariance R in GPFA. Eq A9.
    """
    # Stack all observations and states
    Y = np.hstack([seq['y'][trial] for trial in range(len(seq))])  # Stack all y_:t
    Xsm = np.hstack([seq['xsm'][trial] for trial in range(len(seq))])  # Stack all x_:t
    
    # Get total number of timepoints
    T = np.sum([seq["T"][trial] for trial in range(len(seq))])
    
    if force_diagonal:
        # Compute diagonal elements efficiently
        sum_yytrans = np.sum(Y * Y, axis=1)  # Sum of squared observations
        sum_yall = np.sum(Y, axis=1)  # Sum of observations
        sum_xall = np.sum(Xsm, axis=1)  # Sum of latent states
        
        # Compute yd term
        yd = sum_yall * d
        
        # Compute cross term
        sum_yxtrans = Y @ Xsm.T  # ∑y_:t⟨x_:t⟩'
        term = np.sum((sum_yxtrans - np.outer(d, sum_xall)) * C, axis=1)
        
        # Compute diagonal R (equation A9)
        r = (d**2 + (sum_yytrans - 2*yd - term) / T)
        R = np.diag(r)
        
    else:
        # Compute full covariance matrix
        sum_yytrans = Y @ Y.T  # ∑y_:ty_:t'
        sum_yall = np.sum(Y, axis=1)
        
        # Compute yd terms
        yd = np.outer(sum_yall, d)  # ∑y_:t · d'
        
        # Compute cross term
        sum_yxtrans = Y @ Xsm.T  # ∑y_:t⟨x_:t⟩'
        sum_xall = np.sum(Xsm, axis=1)
        term = (sum_yxtrans - np.outer(d, sum_xall)) @ C.T
        
        # Compute full R (equation A9)
        R = (np.outer(d, d) + 
             (sum_yytrans - yd - yd.T - term) / T)
        
        # Ensure symmetry
        R = (R + R.T) / 2
        
    return R

def grad_K_tau(t1_t2_diff: np.ndarray, tau: float, sigma: float) -> np.ndarray:
    """
    Compute gradient of K with respect to τ. (A10)
    """
    diff_sq = t1_t2_diff**2
    return sigma * (diff_sq / tau**3) * np.exp(-diff_sq / (2 * tau**2))


def grad_E_K(K_inv: np.ndarray, xxT: np.ndarray) -> np.ndarray:
    """
    Compute gradient of E(θ) with respect to K. A10.
    """
    return 0.5 * (-K_inv + K_inv @ xxT @ K_inv)

def objective_tau(log_tau: float, t1_t2_diff: np.ndarray, xxT: np.ndarray, 
                 sigma: float) -> Tuple[float, float]:
    """
    Objective function and gradient for τ optimization. 
    See util/minimize.m and grad_betgam.m
    """
    tau = np.exp(log_tau)
    
    # Compute K and its inverse
    diff_sq = t1_t2_diff**2
    K = sigma * np.exp(-diff_sq / (2 * tau**2)) # Eq 3 page
    K_inv = np.linalg.inv(K)
    
    # Compute gradients (equation A10)
    dE_dK = grad_E_K(K_inv, xxT)
    dK_dtau = grad_K_tau(t1_t2_diff, tau, sigma)
    
    # Compute gradient using trace formula
    grad = np.sum(dE_dK * dK_dtau)  # equivalent to tr(dE_dK' * dK_dtau)
    
    # Convert gradient to log space using chain rule
    grad_log = grad * tau  # d/d(log τ) = τ * d/dτ
    
    # Compute objective value (negative log likelihood)
    obj = -0.5 * (np.log(np.linalg.det(K)) + np.trace(K_inv @ xxT))
    
    return obj, grad_log

def optimize_timescales(seq: np.recarray, x_dim: int, tau: np.ndarray) -> np.ndarray:
    """
    Optimize timescales τᵢ for each latent dimension.
    """
    # Get time points
    T = seq[0].T
    t = np.arange(T)
    t1_t2_diff = t[:, None] - t[None, :]
    
    
    # Optimize for each dimension separately
    for i in range(x_dim):
        # Compute ⟨x'_i:x_i:⟩ for this dimension
        xxT = np.zeros((T, T))
        for trial in range(len(seq)):
            xxT += (seq["Vsm"][trial][i, i, :, :] + 
                   np.outer(seq["xsm"][trial][i, :], seq["xsm"][trial][i, :]))
        xxT /= len(seq)
        
        # Initial guess for log(τ)
        log_tau_init = np.log(1.0)
        
        # Set up optimization
        obj_func = lambda lt: objective_tau(lt, t1_t2_diff, xxT, sigma=1.0)
        
        # Run optimization
        result = minimize(
            fun=lambda lt: obj_func(lt)[0],
            x0=log_tau_init,
            jac=lambda lt: obj_func(lt)[1],
            method='L-BFGS-B'
        )
        
        # Store optimized timescale
        tau.append(np.exp(result.x[0]))
        
    return tau

def m_step(seq: np.recarray, current_params: Dict, **kwargs) -> Dict:
    """Full M-step implementation."""
    # Update C and d
    C, d = m_step_cd(seq, current_params['x_dim'])
    current_params['C'] = C
    current_params['d'] = d
    
    # Update R
    current_params['R'] = m_step_R(seq, C, d)
    
    # Update timescales if needed
    # if current_params['notes']['learnKernelParams']:
    # current_params['tau'] = optimize_timescales(seq, current_params['x_dim'])
        
    return current_params