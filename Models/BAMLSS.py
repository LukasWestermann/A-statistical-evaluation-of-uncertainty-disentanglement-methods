# Bayesian GAMLSS using R's bamlss package via rpy2
# Heteroscedastic likelihood: y ~ Normal(mu(x), sigma(x))
# Decomposition: aleatoric = E[sigma^2], epistemic = Var(mu) across posterior samples

import numpy as np
import pandas as pd
import warnings
import sys
from pathlib import Path

# Add parent directory to path to import utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects.conversion import localconverter
    
    # Suppress R warnings in Python output
    warnings.filterwarnings('ignore')
    
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False
    print("Warning: rpy2 not available. Install with: pip install rpy2")
    print("Also ensure R is installed and available in PATH.")


def _check_rpy2():
    """Check if rpy2 is available, raise error if not."""
    if not RPY2_AVAILABLE:
        raise ImportError(
            "rpy2 is required for BAMLSS. Install with: pip install rpy2\n"
            "Also ensure R is installed and available in PATH."
        )


def _ensure_bamlss_installed():
    """Ensure bamlss package is installed in R."""
    _check_rpy2()
    
    try:
        bamlss = importr('bamlss')
        return bamlss
    except RRuntimeError:
        print("bamlss package not found. Attempting to install...")
        try:
            ro.r('install.packages("bamlss", repos="https://cloud.r-project.org")')
            bamlss = importr('bamlss')
            print("bamlss package installed successfully.")
            return bamlss
        except Exception as e:
            raise RuntimeError(
                f"Failed to install bamlss package. Please install manually in R:\n"
                f"install.packages('bamlss', repos='https://cloud.r-project.org')\n"
                f"Error: {e}"
            )


def fit_bamlss(x_train, y_train, x_grid, n_iter=12000, burnin=2000, thin=10, 
               nsamples=1000, k_mu=20, k_sigma=15, return_raw_arrays=False):
    """
    Fit Bayesian GAMLSS using bamlss package in R.
    
    Parameters:
    -----------
    x_train : np.ndarray, shape (n_train, 1) or (n_train,)
        Training input data
    y_train : np.ndarray, shape (n_train, 1) or (n_train,)
        Training target data
    x_grid : np.ndarray, shape (n_grid, 1) or (n_grid,)
        Grid points for prediction
    n_iter : int
        Number of MCMC iterations
    burnin : int
        Burn-in samples
    thin : int
        Thinning interval
    nsamples : int
        Number of posterior samples for uncertainty decomposition
    k_mu : int
        Basis dimension for mu smooth
    k_sigma : int
        Basis dimension for sigma smooth
    
    Returns:
    --------
    mu_pred : np.ndarray, shape (n_grid,)
        Predictive mean
    ale_var : np.ndarray, shape (n_grid,)
        Aleatoric variance
    epi_var : np.ndarray, shape (n_grid,)
        Epistemic variance
    tot_var : np.ndarray, shape (n_grid,)
        Total variance
    """
    _check_rpy2()
    bamlss = _ensure_bamlss_installed()
    
    # Prepare data - handle both 1D and 2D arrays
    if x_train.ndim == 2:
        x_train = x_train.squeeze()
    if y_train.ndim == 2:
        y_train = y_train.squeeze()
    if x_grid.ndim == 2:
        x_grid = x_grid.squeeze()
    
    # Convert to pandas DataFrame for R
    train_df = pd.DataFrame({'x': x_train, 'y': y_train})
    grid_df = pd.DataFrame({'x': x_grid})
    
    # Convert to R data frames using context manager (replaces deprecated activate())
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_train = ro.conversion.py2rpy(train_df)
        r_grid = ro.conversion.py2rpy(grid_df)
    
    # Store in R environment
    ro.globalenv['r_train'] = r_train
    ro.globalenv['r_grid'] = r_grid
    
    # Define formula in R
    ro.r(f'''
    form <- list(
      y ~ s(x, bs = "ps", k = {k_mu}),
      sigma ~ s(x, bs = "ps", k = {k_sigma})
    )
    ''')
    
    # Fit model
    print(f"Fitting BAMLSS model (this may take a while: {n_iter} iterations)...")
    ro.r(f'''
    fit <- bamlss(
      formula = form,
      family = "gaussian",
      data = r_train,
      sampler = TRUE,
      n.iter = {n_iter},
      burnin = {burnin},
      thin = {thin}
    )
    ''')
    
    # Get posterior samples
    print(f"Extracting {nsamples} posterior samples...")
    # Store nsamples in R environment first
    ro.globalenv['nsamples'] = nsamples
    
    # Try to extract samples - predict() with nsamples should return samples
    # If it doesn't work, we'll extract from MCMC chain directly
    ro.r(f'''
    # First, try predict() with nsamples parameter
    mu_samps_raw <- tryCatch({{
      predict(fit, newdata = r_grid, model = "mu", type = "parameter", nsamples = nsamples)
    }}, error = function(e) {{
      warning("predict() with nsamples failed: ", e$message)
      NULL
    }})
    
    sg_samps_raw <- tryCatch({{
      predict(fit, newdata = r_grid, model = "sigma", type = "parameter", nsamples = nsamples)
    }}, error = function(e) {{
      warning("predict() with nsamples failed: ", e$message)
      NULL
    }})
    
    # Check if we got samples or just means
    n_grid <- length(r_grid$x)
    got_samples <- FALSE
    
    if(!is.null(mu_samps_raw) && !is.null(sg_samps_raw)) {{
      mu_mat <- as.matrix(mu_samps_raw)
      sg_mat <- as.matrix(sg_samps_raw)
      
      # Check if we have the right shape [N x S] or [S x N]
      # If we got [N x 1] or [1 x N], predict() returned means, not samples
      if((nrow(mu_mat) == n_grid && ncol(mu_mat) == nsamples) ||
         (ncol(mu_mat) == n_grid && nrow(mu_mat) == nsamples)) {{
        got_samples <- TRUE
      }} else if(nrow(mu_mat) == n_grid && ncol(mu_mat) == 1) {{
        # Got means, not samples
        got_samples <- FALSE
        print(paste("predict() returned means (shape", nrow(mu_mat), "x", ncol(mu_mat), "), not samples. Extracting from chain..."))
      }} else if(ncol(mu_mat) == n_grid && nrow(mu_mat) == 1) {{
        # Got means transposed
        got_samples <- FALSE
        print(paste("predict() returned means (shape", nrow(mu_mat), "x", ncol(mu_mat), "), not samples. Extracting from chain..."))
      }}
    }}
    
    if(!got_samples) {{
      # predict() didn't return samples - extract from MCMC chain
      print("Extracting samples from MCMC chain...")
      
  # Get samples from fitted model
  # Try samples() function first, then fit$samples
  samps <- tryCatch({{
    samples(fit)
  }}, error = function(e) {{
    if(!is.null(fit$samples)) {{
      fit$samples
    }} else {{
      stop("Cannot extract samples: ", e$message)
    }}
  }})
  
  if(is.null(samps)) {{
    stop("Cannot extract samples from fitted model. fit$samples is NULL")
  }}
  
  # Debug: print structure of samples
  print(paste("Samples structure: class =", class(samps), ", length =", length(samps)))
  if(is.list(samps) && length(samps) > 0) {{
    print(paste("First element class:", class(samps[[1]])))
    if(is.matrix(samps[[1]]) || is.data.frame(samps[[1]])) {{
      print(paste("First element dimensions:", nrow(samps[[1]]), "x", ncol(samps[[1]])))
    }}
  }}
  
  # Check structure of samples - bamlss stores samples as a list of matrices
  # Each element corresponds to a parameter (mu, sigma, etc.)
  # Find the first non-empty element to get chain length
  n_chain <- NULL
  for(i in 1:length(samps)) {{
    if(!is.null(samps[[i]]) && is.matrix(samps[[i]]) && nrow(samps[[i]]) > 0) {{
      n_chain <- nrow(samps[[i]])
      print(paste("Found chain length", n_chain, "from element", i))
      break
    }}
  }}
  
  if(is.null(n_chain) || length(n_chain) == 0 || n_chain == 0) {{
    # Try alternative: check if samples is a data frame or matrix directly
    if(is.matrix(samps) || is.data.frame(samps)) {{
      n_chain <- nrow(samps)
    }} else if(is.list(samps) && length(samps) > 0) {{
      # Try to find any element with rows
      for(i in 1:length(samps)) {{
        if(!is.null(samps[[i]])) {{
          if(is.matrix(samps[[i]]) && nrow(samps[[i]]) > 0) {{
            n_chain <- nrow(samps[[i]])
            break
          }} else if(is.data.frame(samps[[i]]) && nrow(samps[[i]]) > 0) {{
            n_chain <- nrow(samps[[i]])
            break
          }}
        }}
      }}
    }}
    
    if(is.null(n_chain) || length(n_chain) == 0 || n_chain == 0) {{
      stop("Cannot determine chain length from samples. Structure: ", class(samps), ", length: ", length(samps))
    }}
  }}
  
  if(n_chain < nsamples) {{
    warning("Chain has only ", n_chain, " samples, but ", nsamples, " requested. Using all available.")
    n_use <- n_chain
    idx_use <- 1:n_chain
  }} else {{
    n_use <- nsamples
    # Randomly sample nsamples from chain
    idx_use <- sample(n_chain, n_use)
  }}
      
  # Predict for each sample (this is slow but necessary)
  print(paste("Predicting for", n_use, "MCMC samples (this may take a while)..."))
  mu_list <- list()
  sg_list <- list()
  
  for(i in 1:n_use) {{
    # Get the i-th sample index
    idx_i <- idx_use[i]
    
    # Create a temporary fit object with this sample's parameters
    # We need to extract the i-th row from each parameter's samples
    fit_i <- fit
    if(is.list(samps)) {{
      # Extract i-th sample from each parameter
      fit_i$samples <- lapply(samps, function(x) {{
        if(is.matrix(x) && nrow(x) >= idx_i) {{
          x[idx_i, , drop = FALSE]
        }} else if(is.data.frame(x) && nrow(x) >= idx_i) {{
          as.matrix(x[idx_i, , drop = FALSE])
        }} else {{
          x  # Keep as is if we can't index
        }}
      }})
    }}
    
    # Predict for this sample
    mu_pred_i <- predict(fit_i, newdata = r_grid, model = "mu", type = "parameter")
    sg_pred_i <- predict(fit_i, newdata = r_grid, model = "sigma", type = "parameter")
    
    mu_list[[i]] <- as.numeric(mu_pred_i)
    sg_list[[i]] <- as.numeric(sg_pred_i)
    
    if(i %% 100 == 0) {{
      print(paste("  Processed", i, "of", n_use, "samples"))
    }}
  }}
      
      # Combine into matrices [N x S]
      mu_samps <- do.call(cbind, mu_list)
      sg_samps <- do.call(cbind, sg_list)
      
    }} else {{
      # We got samples from predict() - just reshape
      mu_mat <- as.matrix(mu_samps_raw)
      sg_mat <- as.matrix(sg_samps_raw)
      
      if(nrow(mu_mat) == n_grid && ncol(mu_mat) == nsamples) {{
        mu_samps <- mu_mat
      }} else if(ncol(mu_mat) == n_grid && nrow(mu_mat) == nsamples) {{
        mu_samps <- t(mu_mat)
      }} else {{
        stop("Cannot reshape mu_samps: got ", nrow(mu_mat), "x", ncol(mu_mat))
      }}
      
      if(nrow(sg_mat) == n_grid && ncol(sg_mat) == nsamples) {{
        sg_samps <- sg_mat
      }} else if(ncol(sg_mat) == n_grid && nrow(sg_mat) == nsamples) {{
        sg_samps <- t(sg_mat)
      }} else {{
        stop("Cannot reshape sg_samps: got ", nrow(sg_mat), "x", ncol(sg_mat))
      }}
    }}
    
    # Final check
    if(nrow(mu_samps) != n_grid || ncol(mu_samps) != nsamples) {{
      stop("mu_samps shape incorrect: got ", nrow(mu_samps), "x", ncol(mu_samps), ", expected ", n_grid, "x", nsamples)
    }}
    if(nrow(sg_samps) != n_grid || ncol(sg_samps) != nsamples) {{
      stop("sg_samps shape incorrect: got ", nrow(sg_samps), "x", ncol(sg_samps), ", expected ", n_grid, "x", nsamples)
    }}
    ''')
    
    # Convert to numpy arrays
    mu_samps = np.array(ro.r['mu_samps'])
    sg_samps = np.array(ro.r['sg_samps'])
    
    # Debug: print shapes to help diagnose issues
    print(f"Debug: mu_samps shape from R: {mu_samps.shape}, ndim: {mu_samps.ndim}")
    print(f"Debug: sg_samps shape from R: {sg_samps.shape}, ndim: {sg_samps.ndim}")
    print(f"Debug: Expected shape: ({len(x_grid)}, {nsamples})")
    
    # Ensure correct shape [N x S] (N points, S samples)
    # Handle different possible shapes from R
    # First, ensure arrays are at least 2D
    if mu_samps.ndim == 1:
        # If 1D, reshape to [N, S] if size matches
        if mu_samps.size == len(x_grid) * nsamples:
            mu_samps = mu_samps.reshape(len(x_grid), nsamples)
        else:
            raise ValueError(f"Cannot reshape mu_samps from 1D shape {mu_samps.shape} to [N={len(x_grid)}, S={nsamples}]. Size: {mu_samps.size}, Expected: {len(x_grid) * nsamples}")
    elif mu_samps.ndim == 2:
        # If 2D, check orientation and fix if needed
        if mu_samps.shape[0] != len(x_grid):
            if mu_samps.shape[1] == len(x_grid):
                mu_samps = mu_samps.T
            elif mu_samps.size == len(x_grid) * nsamples:
                mu_samps = mu_samps.reshape(len(x_grid), nsamples)
            else:
                raise ValueError(f"Cannot reshape mu_samps from {mu_samps.shape} to [N={len(x_grid)}, S={nsamples}]")
    else:
        # If 3D or higher, try to flatten intelligently
        if mu_samps.size == len(x_grid) * nsamples:
            mu_samps = mu_samps.flatten().reshape(len(x_grid), nsamples)
        else:
            raise ValueError(f"Cannot reshape mu_samps from {mu_samps.shape} to [N={len(x_grid)}, S={nsamples}]")
    
    # Same for sigma samples
    if sg_samps.ndim == 1:
        if sg_samps.size == len(x_grid) * nsamples:
            sg_samps = sg_samps.reshape(len(x_grid), nsamples)
        else:
            raise ValueError(f"Cannot reshape sg_samps from 1D shape {sg_samps.shape} to [N={len(x_grid)}, S={nsamples}]. Size: {sg_samps.size}, Expected: {len(x_grid) * nsamples}")
    elif sg_samps.ndim == 2:
        if sg_samps.shape[0] != len(x_grid):
            if sg_samps.shape[1] == len(x_grid):
                sg_samps = sg_samps.T
            elif sg_samps.size == len(x_grid) * nsamples:
                sg_samps = sg_samps.reshape(len(x_grid), nsamples)
            else:
                raise ValueError(f"Cannot reshape sg_samps from {sg_samps.shape} to [N={len(x_grid)}, S={nsamples}]")
    else:
        if sg_samps.size == len(x_grid) * nsamples:
            sg_samps = sg_samps.flatten().reshape(len(x_grid), nsamples)
        else:
            raise ValueError(f"Cannot reshape sg_samps from {sg_samps.shape} to [N={len(x_grid)}, S={nsamples}]")
    
    # Final check: ensure we have the right shape
    if mu_samps.shape != (len(x_grid), nsamples):
        raise ValueError(f"mu_samps shape {mu_samps.shape} does not match expected ({len(x_grid)}, {nsamples})")
    if sg_samps.shape != (len(x_grid), nsamples):
        raise ValueError(f"sg_samps shape {sg_samps.shape} does not match expected ({len(x_grid)}, {nsamples})")
    
    # Uncertainty decomposition
    mu_mean = np.mean(mu_samps, axis=1)           # E[μ] - predictive mean
    ale_var = np.mean(sg_samps**2, axis=1)        # E[σ²] - aleatoric variance
    epi_var = np.var(mu_samps, axis=1)            # Var[μ] - epistemic variance
    tot_var = ale_var + epi_var                   # Total variance
    
    if return_raw_arrays:
        # Transpose to [S, N] format for consistency with other models
        mu_samps_T = mu_samps.T  # [nsamples, n_grid]
        sigma2_samps_T = (sg_samps ** 2).T  # [nsamples, n_grid]
        return mu_mean, ale_var, epi_var, tot_var, (mu_samps_T, sigma2_samps_T)
    else:
        return mu_mean, ale_var, epi_var, tot_var


def bamlss_predict(x_train, y_train, x_grid, return_raw_arrays=False, **kwargs):
    """
    Wrapper function matching the interface of other models.
    Returns predictions compatible with existing plotting functions.
    
    Parameters:
    -----------
    x_train : np.ndarray, shape (n_train, 1) or (n_train,)
        Training input data
    y_train : np.ndarray, shape (n_train, 1) or (n_train,)
        Training target data
    x_grid : np.ndarray, shape (n_grid, 1) or (n_grid,)
        Grid points for prediction
    return_raw_arrays : bool, default=False
        If True, also return raw (mu_samples, sigma2_samples) arrays
    **kwargs : dict
        Additional arguments passed to fit_bamlss (n_iter, burnin, thin, etc.)
    
    Returns:
    --------
    mu_pred : np.ndarray, shape (n_grid, 1)
        Predictive mean (reshaped to 2D for consistency)
    ale_var : np.ndarray, shape (n_grid,)
        Aleatoric variance
    epi_var : np.ndarray, shape (n_grid,)
        Epistemic variance
    tot_var : np.ndarray, shape (n_grid,)
        Total variance
    (mu_samples, sigma2_samples) : tuple, optional
        Raw arrays if return_raw_arrays=True
    """
    result = fit_bamlss(x_train, y_train, x_grid, return_raw_arrays=return_raw_arrays, **kwargs)
    
    if return_raw_arrays:
        mu_pred, ale_var, epi_var, tot_var, raw_arrays = result
    else:
        mu_pred, ale_var, epi_var, tot_var = result
    
    # Reshape mu_pred to [N, 1] for consistency with other models
    mu_pred = mu_pred.reshape(-1, 1)
    
    if return_raw_arrays:
        return mu_pred, ale_var, epi_var, tot_var, raw_arrays
    else:
        return mu_pred, ale_var, epi_var, tot_var


def fit_bamlss_2d(x_train, z_train, y_train, x_grid, z_grid, 
                  n_iter=12000, burnin=2000, thin=10, 
                  nsamples=1000, k_mu=10, k_sigma=10, return_raw_arrays=False):
    """
    Fit Bayesian GAMLSS using bamlss package in R with 2D input (X and Z).
    
    Parameters:
    -----------
    x_train : np.ndarray, shape (n_train,)
        First dimension of training input (X)
    z_train : np.ndarray, shape (n_train,)
        Second dimension of training input (Z)
    y_train : np.ndarray, shape (n_train,)
        Training target data
    x_grid : np.ndarray, shape (n_grid,)
        First dimension grid points for prediction
    z_grid : np.ndarray, shape (n_grid,)
        Second dimension grid points for prediction
    n_iter, burnin, thin, nsamples, k_mu, k_sigma : int
        MCMC and model parameters
    return_raw_arrays : bool
        If True, return raw (mu_samples, sigma2_samples)
    
    Returns:
    --------
    Same as fit_bamlss but for 2D input
    """
    _check_rpy2()
    bamlss = _ensure_bamlss_installed()
    
    # Prepare data - flatten all inputs
    x_train = np.asarray(x_train).flatten()
    z_train = np.asarray(z_train).flatten()
    y_train = np.asarray(y_train).flatten()
    x_grid = np.asarray(x_grid).flatten()
    z_grid = np.asarray(z_grid).flatten()
    
    n_grid = len(x_grid)
    
    # Convert to pandas DataFrame for R
    train_df = pd.DataFrame({'x': x_train, 'z': z_train, 'y': y_train})
    grid_df = pd.DataFrame({'x': x_grid, 'z': z_grid})
    
    # Convert to R data frames
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_train = ro.conversion.py2rpy(train_df)
        r_grid = ro.conversion.py2rpy(grid_df)
    
    ro.globalenv['r_train'] = r_train
    ro.globalenv['r_grid'] = r_grid
    
    # Define formula with both x and z using tensor product smooth
    ro.r(f'''
    form <- list(
      y ~ s(x, bs = "ps", k = {k_mu}) + s(z, bs = "ps", k = {k_mu}) + ti(x, z, bs = "ps", k = {min(k_mu, 5)}),
      sigma ~ s(x, bs = "ps", k = {k_sigma}) + s(z, bs = "ps", k = {k_sigma})
    )
    ''')
    
    print(f"Fitting BAMLSS 2D model (this may take a while: {n_iter} iterations)...")
    ro.r(f'''
    fit <- bamlss(
      formula = form,
      family = "gaussian",
      data = r_train,
      sampler = TRUE,
      n.iter = {n_iter},
      burnin = {burnin},
      thin = {thin}
    )
    ''')
    
    print(f"Extracting {nsamples} posterior samples...")
    ro.globalenv['nsamples'] = nsamples
    ro.globalenv['n_grid'] = n_grid
    
    # Use simple prediction approach - get mean predictions
    # For 2D case, we simplify by getting point estimates
    ro.r('''
    mu_pred <- predict(fit, newdata = r_grid, model = "mu", type = "parameter")
    sg_pred <- predict(fit, newdata = r_grid, model = "sigma", type = "parameter")
    
    # Get samples from chain for uncertainty
    samps <- samples(fit)
    
    # Find chain length
    n_chain <- 0
    for(i in 1:length(samps)) {
      if(!is.null(samps[[i]]) && is.matrix(samps[[i]])) {
        n_chain <- nrow(samps[[i]])
        break
      }
    }
    
    if(n_chain < nsamples) {
      n_use <- n_chain
    } else {
      n_use <- nsamples
    }
    
    idx_use <- sample(n_chain, n_use)
    
    # Simple approach: predict for each sample
    mu_list <- list()
    sg_list <- list()
    
    for(i in 1:n_use) {
      idx_i <- idx_use[i]
      fit_i <- fit
      fit_i$samples <- lapply(samps, function(x) {
        if(is.matrix(x) && nrow(x) >= idx_i) {
          x[idx_i, , drop = FALSE]
        } else {
          x
        }
      })
      
      mu_pred_i <- predict(fit_i, newdata = r_grid, model = "mu", type = "parameter")
      sg_pred_i <- predict(fit_i, newdata = r_grid, model = "sigma", type = "parameter")
      
      mu_list[[i]] <- as.numeric(mu_pred_i)
      sg_list[[i]] <- as.numeric(sg_pred_i)
      
      if(i %% 100 == 0) {
        print(paste("  Processed", i, "of", n_use, "samples"))
      }
    }
    
    mu_samps <- do.call(cbind, mu_list)
    sg_samps <- do.call(cbind, sg_list)
    ''')
    
    mu_samps = np.array(ro.r['mu_samps'])
    sg_samps = np.array(ro.r['sg_samps'])
    
    # Ensure correct shape [N x S]
    if mu_samps.ndim == 1:
        mu_samps = mu_samps.reshape(n_grid, -1)
    if sg_samps.ndim == 1:
        sg_samps = sg_samps.reshape(n_grid, -1)
    
    if mu_samps.shape[0] != n_grid:
        if mu_samps.shape[1] == n_grid:
            mu_samps = mu_samps.T
            sg_samps = sg_samps.T
    
    # Uncertainty decomposition
    mu_mean = np.mean(mu_samps, axis=1)
    ale_var = np.mean(sg_samps**2, axis=1)
    epi_var = np.var(mu_samps, axis=1)
    tot_var = ale_var + epi_var
    
    if return_raw_arrays:
        mu_samps_T = mu_samps.T
        sigma2_samps_T = (sg_samps ** 2).T
        return mu_mean, ale_var, epi_var, tot_var, (mu_samps_T, sigma2_samps_T)
    else:
        return mu_mean, ale_var, epi_var, tot_var


def bamlss_predict_2d(x_train, z_train, y_train, x_grid, z_grid, return_raw_arrays=False, **kwargs):
    """
    BAMLSS prediction with 2D input for OVB experiments.
    
    Parameters:
    -----------
    x_train : np.ndarray
        First dimension of training input
    z_train : np.ndarray
        Second dimension of training input (omitted variable)
    y_train : np.ndarray
        Training targets
    x_grid : np.ndarray
        First dimension of prediction grid
    z_grid : np.ndarray
        Second dimension of prediction grid
    return_raw_arrays : bool
        If True, return raw sample arrays
    **kwargs : dict
        Additional arguments passed to fit_bamlss_2d
    
    Returns:
    --------
    Same as bamlss_predict
    """
    result = fit_bamlss_2d(x_train, z_train, y_train, x_grid, z_grid, 
                           return_raw_arrays=return_raw_arrays, **kwargs)
    
    if return_raw_arrays:
        mu_pred, ale_var, epi_var, tot_var, raw_arrays = result
    else:
        mu_pred, ale_var, epi_var, tot_var = result
    
    mu_pred = mu_pred.reshape(-1, 1)
    
    if return_raw_arrays:
        return mu_pred, ale_var, epi_var, tot_var, raw_arrays
    else:
        return mu_pred, ale_var, epi_var, tot_var

