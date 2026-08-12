# Models package for uncertainty disentanglement methods

try:
    from Models.BAMLSS import fit_bamlss, bamlss_predict
    __all__ = ['fit_bamlss', 'bamlss_predict']
except Exception as e:
    # BAMLSS depends on R/rpy2, which may not be set up in this environment.
    # Don't let that block importing the other (PyTorch-based) models.
    print(f"Warning: BAMLSS unavailable ({e}). Other Models submodules still work.")
    __all__ = []

