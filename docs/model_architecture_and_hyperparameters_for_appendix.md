# Model architectures, software stack, and hyperparameters (appendix source text)

You can paste this document into Claude (or your thesis appendix) as a **transparent specification** of how predictive models are built and trained in this repository. Paths refer to the codebase under `Models/` (regression) and `utils/classification_models.py` (classification). **Numerical defaults** below match the main **regression experiment drivers** in `utils/*_experiments.py` (e.g. OOD, sample size, noise level) unless noted; **classification notebooks** sometimes override `K`, `mc_samples`, or `epochs`—check `docs/classification_experiments_specifications.md` for notebook-specific training fields.

---

## 1. Software stack

| Component | Role |
|-----------|------|
| **Python** | Orchestration, NumPy arrays, experiment loops |
| **PyTorch** (`torch`, `torch.nn`, `torch.optim`) | MC Dropout, Deep Ensemble, and optimization; automatic differentiation |
| **Pyro** (`pyro`, `pyro.distributions`, `pyro.infer.MCMC`, `NUTS`, `Predictive`) | Bayesian neural networks: prior specification and NUTS sampling (**CPU** for MCMC in this project) |
| **NumPy** | Data arrays, stacking posterior / MC draws |
| **R + `bamlss` + `rpy2`** | Bayesian GAMLSS: `Models/BAMLSS.py` fits models in R and returns posterior draws to Python |

Device selection for neural training uses `utils/device.py` (`get_device()`): CUDA when available, else CPU. **BNN MCMC is forced to CPU** in `Models/BNN.py` for Pyro stability.

---

## 2. Regression: shared heteroscedastic Gaussian predictive head

All neural regression baselines predict a **Gaussian** conditional for $y \mid x$ with mean $\mu(x)$ and variance $\sigma^2(x) > 0$.

- **Mean head:** linear layer → scalar $\mu$.
- **Variance head:** linear layer → **Softplus** → $\sigma > 0$; **variance** stored as $\sigma^2$ (with small floor `1e-6` where applicable).
- **Trunk:** two hidden layers, width **32**, **ReLU** activations, input dimension **1** for scalar $x$ (OVB uses `input_dim=2` when $(X,Z)$ is passed in specialized code).

**Uncertainty decomposition** (Gal-style, over $M$ or $K$ forward draws):

- $\sigma^2_{\mathrm{ale}}(x) = \frac{1}{M}\sum_m \sigma^{2,(m)}(x)$
- $\sigma^2_{\mathrm{epi}}(x) = \mathrm{Var}_m\bigl(\mu^{(m)}(x)\bigr)$
- $\sigma^2_{\mathrm{tot}} = \sigma^2_{\mathrm{ale}} + \sigma^2_{\mathrm{epi}}$

---

## 3. Monte Carlo Dropout (regression)

**Source:** `Models/MC_Dropout.py` — class `MCDropoutRegressor`.

**Architecture**

- `Linear(1 → 32) → ReLU → Dropout(p) → Linear(32 → 32) → ReLU → Dropout(p)`  
- Heads: `mu_head: Linear(32 → 1)`; `sigma_head: Linear(32 → 1) → Softplus → +eps`, variance $= \sigma^2$.

**Dropout** probability default **$p = 0.25$** on both hidden layers.

**Training**

- Optimizer: **Adam**, learning rate **$10^{-3}$**, default **500** epochs, batch size **32**.
- Loss: **$\beta$-NLL** with **$\beta = 0.5$** (`loss_type='beta_nll'`): Gaussian NLL weighted by $(\sigma^2)^\beta$ with **stop-gradient** on $\sigma^2$ in the weight (see `beta_nll()` in the same file). Plain Gaussian NLL is available as `loss_type='nll'`.

**Prediction / UQ**

- Model left in **`train()`** mode so dropout stays active.
- **$M$** stochastic forward passes (default **$M = 100$** in `mc_dropout_predict`, configurable as `mc_samples` in experiment drivers).
- Stacks `mu` and `var` per pass → decomposition as in §2.

**Reproducibility**

- Experiment scripts typically call **`np.random.seed(seed)`** and **`torch.manual_seed(seed)`** with **`seed = 42`** before building the `DataLoader` and model (e.g. `utils/ood_experiments.py`). MC stochasticity at prediction time remains unless you fix additional RNG state.

**Note:** In **Deep Ensemble** regression runs in `utils/ood_experiments.py`, inputs are **z-normalized** using training $x$ statistics before training and grid evaluation; **MC Dropout** on the same OOD path uses **raw** $x$ in the cited driver—intentionally asymmetric between baselines in that file.

---

## 4. Deep Ensemble (regression)

**Source:** `Models/Deep_Ensemble.py` — class `BaselineRegressor`.

**Architecture**

- **No dropout:** `Linear(1 → 32) → ReLU → Linear(32 → 32) → ReLU` → same $\mu$ / $\sigma$ heads as MC Dropout (Softplus variance).

**Training**

- **$K$** separate networks, default **$K = 20$** in `run_deep_ensemble_ood_experiment` (function default in `utils/ood_experiments.py`). The generic `train_ensemble_deep(..., K=30)` default in `Deep_Ensemble.py` is **overridden** by callers—use the experiment argument.
- Same optimizer and schedule as MC Dropout: **Adam**, **lr $10^{-3}$**, **500** epochs, batch **32**, **$\beta$-NLL** with **$\beta = 0.5$**.
- **Member seeds:** `member_seed = base_seed + 1000 + k` with **`base_seed = 42`** in `Deep_Ensemble.py` (per-member initialization and training shuffle).
- Members may train **in parallel** (`ThreadPoolExecutor`) when `parallel=True`.

**Prediction / UQ**

- Each member in **`eval()`** mode (deterministic forward).
- One $(\mu_k, \sigma^2_k)$ per member → stack as $K$ “samples” → same decomposition formulas as §2 with $M=K$.

---

## 5. Bayesian neural network (regression, Pyro)

**Source:** `Models/BNN.py`.

**Architecture (probabilistic)**

- Two hidden layers, ReLU, hidden width default **$H = 16$** (`hidden_width`).
- **Priors:** independent **Normal$(0, \texttt{weight\_scale}^2)$** on all weight and bias tensors, with `weight_scale=1.0` default; shapes match layer dimensions (`W1`, `b1`, `W2`, `b2`, `W_mu`, `b_mu`, `W_rho`, `b_rho`).
- **Likelihood:** $y_i \sim \mathcal{N}(\mu_\theta(x_i), \sigma_\theta(x_i))$ with $\sigma_\theta = \mathrm{softplus}(\rho_\theta(x)) + 10^{-6}$.

**Inference**

- **NUTS** kernel, **`target_accept_prob = 0.8`**.
- Defaults: **`warmup_steps = 200`**, **`num_samples = 200`** posterior draws, **`num_chains = 1`**.
- **`pyro.set_rng_seed(seed)`** when `seed` is passed to `run_nuts` / `train_bnn` (note: module also calls `pyro.set_rng_seed(0)` at import—experiments should pass explicit `seed` for full reproducibility).

**Prediction**

- **`Predictive`** over the trained posterior to draw $\mu$ and $\sigma$ per grid point; decomposition uses **`decompose_uncertainty`**: $\mathbb{E}[\sigma^2]$ and $\mathrm{Var}(\mu)$ across posterior draws.

**Compute**

- **CPU only** for MCMC and prediction (`cpu_device`).

---

## 6. BAMLSS (regression, R backend)

**Source:** `Models/BAMLSS.py`.

**Role**

- Bayesian **GAMLSS** in **R** (`bamlss` package) via **`rpy2`**; heteroscedastic Gaussian (or extended) structured additive predictors for $\mu(x)$ and $\log\sigma(x)$ depending on the fitted formula.

**Typical MCMC settings (defaults in `fit_bamlss`)**

- **`n_iter = 12000`**, **`burnin = 2000`**, **`thin = 10`**.
- Prediction uses a large number of retained draws (e.g. **`nsamples = 1000`** in `bamlss_predict` call sites in experiment code—confirm per run).

**Reproducibility**

- Controlled by **R / bamlss** RNG; Python side passes data through `rpy2`. Document the R seed if you set it in a notebook.

---

## 7. Classification (2D input): MC Dropout, Deep Ensemble, BNN

**Source:** `utils/classification_models.py` (separate from regression `Models/`).

**Shared trunk geometry**

- **Input dimension 2** (default), **$C$** classes (default **3** for blobs; **2** for ring OOD).
- **MC Dropout (IT):** `Linear(2→32) → ReLU → Dropout(p) → Linear(32→32) → ReLU → Dropout(p) → Linear(32→C)` → **logits**; training loss **cross-entropy**.
- **MC Dropout (GL):** same trunk with dropout; **$\mu$ head** `Linear(32→C)` and **$\sigma^2$ head** `Linear(32→C) → Softplus`, variance $= \sigma^2 + 10^{-6}$. Training minimizes CE on **MC softmax**: sample $z \sim \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$, softmax, average over **100** inner samples default (`sampling_softmax_torch`).
- **Deep Ensemble (IT / GL):** same widths, **no dropout**; GL training uses the same sampling-softmax CE.

**Defaults in `classification_models.py`**

| Parameter | Default |
|-----------|---------|
| Dropout $p$ | 0.25 |
| Epochs | 300 |
| Adam lr | $10^{-3}$ |
| Batch size | 32 |
| MC passes at prediction (IT / GL) | 100 |
| Inner softmax samples (GL train) | 100 |
| Ensemble size $K$ | 10 |
| BNN hidden width | 32 |
| BNN prior scale | 1.0 |
| BNN NUTS | target accept 0.8, warmup 200, samples 200, chains 1 |

**Notebook overrides (examples)**

- `Classification_Sample_Size.ipynb` / `Classification_Label_Noise.ipynb`: **`K = 5`**, **`mc_samples = 50`**.
- `Classification_RCD.ipynb` / `Classification_Ring_OOD.ipynb`: **`K = 20`**, **`mc_samples = 100`**.

**BNN classification**

- Pyro models `bnn_classification_it` / `bnn_classification_gl` with two ReLU hidden layers, width **32**, Gaussian priors scale **1.0**; GL adds $\rho$ head for logit scales. Same NUTS defaults as regression BNN where not overridden.

**Reproducibility**

- Notebooks set **`np.random.seed(42)`** and **`torch.manual_seed(42)`**; BNN training uses **`pyro.set_rng_seed(seed)`** with cfg `seed` (typically 42).

---

## 8. Uncertainty outputs used downstream (short)

- **Regression:** variance decomposition on $(\mu^{(m)}, \sigma^{2,(m)})$; entropies from `utils/entropy_uncertainty.py` (analytical / numerical / etc.) when enabled in experiments.
- **Classification IT:** member softmax tensors → `it_uncertainty` (TU, AU, EU + per-map min–max norms).
- **Classification GL:** pooled $\bar\mu$, $\sigma^2_{\mathrm{ale}}$, $\sigma^2_{\mathrm{epi}}$ → MC softmax → entropies (`gl_uncertainty`).

---

## 9. One-paragraph “methods blurb” for an appendix

*We implement four probabilistic baselines on shared synthetic tasks. MC Dropout uses a two-hidden-layer ReLU network (width 32) with dropout probability 0.25 before the output heads; at test time we keep dropout active and average $M$ stochastic forward passes. The predictive head outputs a Gaussian mean and a positive standard deviation via Softplus; we train with Adam ($10^{-3}$ learning rate) for 500 epochs (regression) or 300 epochs (classification) using $\beta$-NLL ($\beta=0.5$, regression) or cross-entropy / sampling-softmax (classification). Deep ensembles use the same trunk **without** dropout, train $K$ networks with distinct random seeds, and aggregate predictions across members. Bayesian neural networks use Pyro: independent Gaussian priors on all weights, two ReLU hidden layers (width 16 for regression, 32 for classification), heteroscedastic Gaussian likelihood, and NUTS (target acceptance 0.8, 200 warmup steps, 200 samples, single chain) on CPU. BAMLSS fits a Bayesian GAMLSS in R via rpy2 with 12{,}000 iterations, 2{,}000 burn-in, and thinning 10 unless otherwise specified. Global seeds are set to 42 for NumPy and PyTorch at experiment start; ensemble members use deterministic offsets of the base seed. Full layer definitions appear in `Models/MC_Dropout.py`, `Models/Deep_Ensemble.py`, `Models/BNN.py`, `Models/BAMLSS.py`, and `utils/classification_models.py`.*

---

*This file is meant as **source material** for an appendix; align any sentence with the exact notebook or `utils/*_experiments.py` call you cite in the thesis.*
