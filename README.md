# Uncertainty disentanglement — experiments, outputs, and moment-matched entropy

This repository holds regression (and some classification) experiments, saved model outputs, and plotting / table export scripts. The layout under `results/` mirrors **experiment type** (e.g. `ood`, `sample_size`, `noise_level`). Each experiment typically has **`outputs/`** (heavy artefacts), **`statistics/`** (summaries), and **`plots/`** (figures from the notebooks or utilities).

---

## Environment setup

**Recommended:** use a **Conda** environment (especially on **Windows**, where mixing system Python, PyTorch, and `rpy2` is more fragile than inside an isolated env).

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
2. Create and activate an environment (Python 3.10 or 3.11 works well with the pinned stack):

   ```bash
   conda create -n uncertainty-disent python=3.11 -y
   conda activate uncertainty-disent
   ```

3. From the **repository root**, install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Dependencies are listed in **`requirements.txt`** (NumPy, SciPy, pandas, Jupyter, **PyTorch**, **Pyro**, `rpy2`, OpenPyXL, pytest, etc.). If `pip` fails on a heavy package (e.g. PyTorch), install that piece with Conda first and then run `pip install -r requirements.txt` for the rest.

4. **Jupyter:** register the kernel if you want this env in the notebook picker:

   ```bash
   python -m ipykernel install --user --name uncertainty-disent --display-name "Python (uncertainty-disent)"
   ```

5. **BAMLSS (`Models/BAMLSS.py`):** requires **R** on your PATH and the R package **`bamlss`**. With `rpy2` installed, the code can prompt or you can run in R: `install.packages("bamlss", repos="https://cloud.r-project.org")`.

6. **GPU (optional):** PyTorch will use CUDA if a matching build is installed; BNN (**Pyro MCMC**) in this repo is intended to run on **CPU** (see `Models/BNN.py`).

---

## Primary workflow: notebooks

The intended way to (re)generate end-to-end results is still to run the notebooks under **`Experiments/`**, for example:

| Notebook | Typical `results/` subtree |
|----------|----------------------------|
| `OOD.ipynb` | `results/ood/` |
| `Sample Size.ipynb` | `results/sample_size/` |
| `Noise_Level.ipynb` / `Noise Level.ipynb` | `results/noise_level/` |
| `Undersampling.ipynb` | `results/undersampling/` |
| `OVB_regression.ipynb` | `results/ovb/` |
| Classification notebooks | `results/classification/...` |

In practice, each notebook is a lightweight **wrapper**:

1. Set experiment parameters in the notebook cell(s) (model hyperparameters, experiment ranges, seeds, run flags). The default paramters are already set to the ones used in the thesis.
2. Run the notebook (typically Run All, or at least all cells in order).
3. Then use the saved artifacts under `results/...` for downstream analysis. For quick ad hoc analysics directly inspect the output printed in the notebook.

After a run, you usually get:

- **`results/<experiment>/outputs/...`** — `*raw_outputs*.npz` (predictive samples, grids, metadata). These are the **canonical saved predictions**; recomputation scripts read them.
- **`results/<experiment>/statistics/...`** — CSV / Excel summaries produced by the experiment code (variance-based metrics, and entropy-based metrics **using whatever `entropy_method` the notebook/run used at the time**).
- **`results/<experiment>/plots/...`** — figures written directly from the experiment.

You can run a single notebook or several; paths are fixed by experiment name, not by “run order”.

*How models are trained, which losses are used, and how sampling feeds variance/entropy decompositions is described in **Appendix A** at the end of this file.*

---

## Modular structure (quick map)

The code is intentionally split so notebooks stay short and reusable:

- **Notebooks (`Experiments/*.ipynb`)**: parameter entry points and orchestration (what to run).
- **Experiment wrappers (`utils/*_experiments.py`)**: loop over settings (function type, noise type, tau/pct/OOD ranges), call model train/predict code, aggregate metrics, and trigger saving.
- **Model implementations (`Models/*.py`)**: actual regression model definitions, training loops, predictive sampling functions, and loss logic.
- **Common utilities (`utils/`)**: entropy decomposition, metrics, plotting, and save-path helpers.
- **Offline scripts (`scripts/*.py`)**: post-processing from existing outputs (recompute entropy from npz, build summary panels, export LaTeX tables).

So the normal workflow is:
`Notebook parameters -> utils experiment runner -> Models/utilities -> results/outputs + statistics + plots -> optional scripts for recompute/exports`.

---

## Moment-matched entropy: default in code vs. older runs

**Current default in the Python runners** (e.g. OOD and noise-level experiment functions in `utils/ood_experiments.py`, `utils/noise_level_experiments.py`) is:

- `entropy_method='moment_matched'`

That dispatches to **`entropy_uncertainty_analytical_moment_matched`** in **`utils/entropy_uncertainty.py`**: closed-form aleatoric/epistemic/total entropy from member means and variances (total uncertainty matched to a Gaussian with variance \(\mathbb{E}[\sigma^2] + \mathrm{Var}(\mu)\)).

**Why a separate “recomputation” path exists:** A small bug in the original implementation of the variance used to compute the total uncertainty estimate in the differential entropy case went unnoticed until after many models had already been trained. Crucially, the error affected only the downstream entropy computation — the model outputs themselves were unaffected. Since re-training all models solely to refresh the entropy columns would have been computationally prohibitive, the entropy estimates were instead recomputed directly from the stored model-output .npz files. The batch script therefore **recomputes moment-matched entropy from existing `*raw_outputs*.npz`**, whcih were not affected by the wrong entropy formula, and writes **parallel** statistics and plots under a dedicated tree (below). New notebook runs, with the current corrected defaults, should already write consistent moment-matched entropy into the normal experiment statistics; the batch remains useful for **refreshing plots/tables from old npz** without rerunning training.

---

## Batch moment-matched recomputation (from `raw_outputs` only)

**Script:** `scripts/recompute_entropy_moment_matched_batch_from_npz.py`

**What it does:** Loads `mu_samples` / `sigma2_samples` (and grid metadata) from each npz, applies **`entropy_uncertainty_analytical_moment_matched`** (same idea as the live experiments), computes regional summaries and panel-style figures depending on the experiment.

**Input roots** (by `--experiments` key; each then scans `homoscedastic|heteroscedastic` × `linear|sin` where applicable):

| Key | Npz root (under project) |
|-----|---------------------------|
| `ood` | `results/ood/outputs/ood/` |
| `sample_size` | `results/sample_size/outputs/sample_size/` |
| `noise_level` | `results/noise_level/outputs/noise_level/` |
| `undersampling` | `results/undersampling/outputs/undersampling/` |
| `ovb` | `results/ovb/` (OVB-specific npz naming) |

**Typical invocation:**

```bash
python scripts/recompute_entropy_moment_matched_batch_from_npz.py --batch --experiments ood,sample_size,noise_level
```

Subset examples: `--experiments noise_level` only, or `--models Deep_Ensemble,BAMLSS` (comma-separated stem tags). See `--help` for `--grid-stride`, `--out-root`, `--ood-range`, etc.

**Output root (default):** `results/entropy_recomputed_moment_matched_batch/`

- **`statistics/<experiment>/...`** — per-model Excel workbooks (and OOD-style three-region tables where relevant).
- **`plots/<experiment>/...`** — 2×4 / 1×4 style panels and related figures.
- **`tables/`** — aggregated workbook (e.g. multi-sheet summary by experiment key).

This tree is **only** from the script; it does not replace notebook outputs. It is the **script-side** mirror used for thesis figures and tables when you want numbers strictly tied to moment-matched entropy **without** rerunning notebooks.

---

## Consolidated summary panels (`summary_panels_consolidated`)

**Script:** `scripts/plot_summary_panels_consolidated.py`

**Purpose:** One place to regenerate **sample-size** and **noise-level** summary line panels:

- **Variance** curves: loaded from the **same CSV statistics** as the older `plot_*_summary_4x2.py` helpers (under `results/sample_size/statistics/`, `results/noise_level/statistics/`, etc.).
- **Entropy** curves: **only** from moment-matched batch Excel under **`results/entropy_recomputed_moment_matched_batch/statistics/`** (via `utils/moment_matched_summary_loaders.py`). If no workbook exists for a condition, that entropy panel is skipped (check console `[skip]` lines).

**Default output:** `results/summary_panels_consolidated/sample_size/` and `.../noise_level/` (flat PNGs named with `linear|sin` and `homoscedastic|heteroscedastic`).

```bash
python scripts/plot_summary_panels_consolidated.py
# Optional:
#   --out <path>   --moment-matched-stats <path>   --noise-distribution normal
```

**Order of operations if you care about entropy lines:** (1) notebooks or existing npz → (2) `recompute_entropy_moment_matched_batch_from_npz.py` for the experiments you need → (3) `plot_summary_panels_consolidated.py`.

---

## Other small export scripts (reference only)

These are convenient for LaTeX, Markdown, CSV, or appendix tables; they do not change training. (Plotting and entropy recomputation live in other `scripts/*.py` helpers documented above.)

### OOD (regression)

- **`scripts/export_ood_latex_tables.py`** — OOD table from latest `results/ood/outputs/ood/.../*raw_outputs*.npz` (moment-matched entropy + variance analogue on the same grid).
- **`scripts/build_ood_overview_tables.py`** — overview CSV / appendix LaTeX from `results/ood/statistics/ood/...` ID/OOD CSVs.

### Sample size and noise level (regression)

- **`scripts/export_sample_size_correlation_moment_entropy_latex.py`** — sample-size correlation table (variance CSV + moment-matched batch Excel).
- **`scripts/export_sample_size_correlation_var_entropy_latex.py`** — one LaTeX table with variance-side ρ and moment-matched entropy-side ρ side by side.
- **`scripts/export_noise_level_correlation_moment_entropy_latex.py`** — noise-level (per τ) LaTeX: correlations and MSE from variance CSVs + moment-matched batch workbooks.
- **`scripts/export_sample_size_summary_table_llm.py`** — Markdown/CSV (and optional LaTeX + small overview figures) comparing variance vs moment-matched entropy summaries.
- **`scripts/export_noise_level_summary_table_llm.py`** — same idea for noise level (per distribution, default `normal`).
- **`scripts/export_moment_matched_sample_noise_tables_md.py`** — compact Markdown + LaTeX from `results/entropy_recomputed_moment_matched_batch/statistics/` for sample-size and noise-level moment-matched workbooks.
- **`scripts/build_sample_size_mse_latex_tables.py`** — MSE-only LaTeX from `results/sample_size/statistics/...` variance CSVs.
- **`scripts/build_noise_level_latex_tables.py`** — noise-level overview LaTeX from `results/noise_level/statistics/` (combined/per-model Excel paths as documented in the script).

### Classification

- **`scripts/export_label_noise_accuracy_correlation_latex.py`** — label-noise: accuracy and AU–EU correlation from `results/classification/label_noise/statistics/.../*_eta_summary.xlsx`.
- **`scripts/export_sample_size_accuracy_correlation_latex.py`** — sample-size classification: same metrics from `*_N_train_summary.xlsx`.
- **`scripts/export_ring_ood_accuracy_correlation_latex.py`** — ring OOD: ring vs gap accuracy and AU–EU correlations from `*_ring_ood_summary.xlsx`.

### Undersampling

- **`scripts/export_undersampling_overview_latex.py`** — variance from Excel summaries + moment-matched entropy from latest `raw_outputs` npz; Markdown / LaTeX under `results/undersampling/tables/`.
- **`scripts/build_undersampling_overview_tables.py`** — wider appendix LaTeX from `results/undersampling/statistics/undersampling/...` Excel only (entropy + variance blocks per region).

### OVB (omitted-variable sweep)

- **`scripts/export_ovb_omitted_correlation_mse_latex.py`** — correlation and MSE tables tying legacy OVB sweep Excel to moment-matched batch statistics (same slicing as `plot_ovb_sweep_summary_panels`).

---

## Routine (when you change code or refresh figures)

Use this as a **short checklist** without rereading the whole README:

1. **Run the relevant `Experiments/*.ipynb`** (or all that you care about). Confirm new `outputs/` / `statistics/` / `plots/` under the matching `results/<experiment>/` folder.
2. **If you need moment-matched entropy panels or tables from existing npz** (no retrain): run  
   `python scripts/recompute_entropy_moment_matched_batch_from_npz.py --batch --experiments <keys>`  
   with the keys you changed (`noise_level`, `sample_size`, `ood`, …).
3. **If you need the flat consolidated summary PNGs** (`results/summary_panels_consolidated/`): run  
   `python scripts/plot_summary_panels_consolidated.py`  
   after step 2 for entropy side; variance side only needs CSV statistics from step 1.
4. **Optional:** run the small **export_*.py** / **build_*_tables.py** helpers in [Other small export scripts](#other-small-export-scripts-reference-only) for LaTeX or summary snippets when drafting the thesis.

The “source of truth” for **what actually happened in a given run** remains the notebook + its `results/<experiment>/` tree. The moment-matched **batch** directory is an explicit, reproducible **recompute-from-npz** layer on top of `raw_outputs`, aligned with the **current** default entropy definition in code.

---

## Utilities worth knowing (high level)

| Module | Role |
|--------|------|
| `utils/entropy_uncertainty.py` | Analytical / numerical / **moment-matched** entropy decomposition; `entropy_uncertainty_by_method` string dispatch. |
| `utils/knn_entropy_regression.py` | Reading npz, OOD masks, **`compute_moment_matched_grid_result`**, regional normalized stats for batch recomputation and some panels. |
| `utils/moment_matched_summary_loaders.py` | Load per-model DataFrames from batch Excel for summary panels. |
| `utils/regression_summary_panels.py` | Shared panel builders (4×2, 1×4 AU/EU, correlation-only) used by notebooks and scripts. |
| `utils/ood_experiments.py` / `noise_level_experiments.py` | Experiment loops, variance + entropy statistics, saving under `results/`. |
| `utils/results_save.py` | `save_model_outputs`, statistics and plot paths (`outputs_dir`, `stats_dir`, `plots_dir`). |

**Training, losses, and predictive sampling** live mainly under **`Models/`** (regression) and **`utils/metrics.py`**; see **Appendix A**.

For deeper behaviour (normalization, OOD ranges, column names), follow calls from the script or notebook into these modules rather than duplicating full detail here.

---

## Appendix A — Training pipeline (models, losses, sampling, decompositions)

This appendix describes **how predictions are produced**, not where CSVs are saved. Regression toy experiments **import from `Models/`** inside `utils/*_experiments.py` (e.g. `ood_experiments.py`, `noise_level_experiments.py`). Classification uses **`utils/classification_models.py`** (separate architectures and training helpers).

### A.1 `Models/` — regression architectures and training

| File | What it defines | Training objective | Inference / sampling |
|------|-----------------|--------------------|----------------------|
| **`Models/MC_Dropout.py`** | `MCDropoutRegressor`: two hidden layers (32, ReLU), dropout on both; heads for **μ** (linear) and **σ** (Softplus → variance). | **`gaussian_nll`** or **`beta_nll`** (β-NLL with stop-grad weighting on σ²); optimized with Adam in **`train_model`**. | **`mc_dropout_predict`**: model stays in **`train()`** mode so dropout stays on; **`M`** stochastic forward passes; stacks `μ_m`, `σ²_m`. |
| **`Models/Deep_Ensemble.py`** | `BaselineRegressor`: same heteroscedastic Gaussian head idea, **no** dropout. **`train_ensemble_deep`** trains **`K`** members (different seeds; optional thread pool). | Reuses **`gaussian_nll` / `beta_nll`** from `MC_Dropout.py` via import. | **`ensemble_predict_deep`**: each member in **`eval()`**; one deterministic forward per member → same stacking as MC Dropout. |
| **`Models/BNN.py`** | Pyro **`bnn_model`**: priors on weights, heteroscedastic likelihood **y ~ Normal(μ(x), σ(x))**. | **NUTS** MCMC (`pyro.infer.MCMC`); see module for `num_samples`, warmup, etc. | **`Predictive`** (and helpers in the file) draw **posterior samples** of weights, then push the grid through the net to get per-sample **μ** and **σ** (then σ²) on each point. Runs on **CPU** by design in this repo. |
| **`Models/BAMLSS.py`** | Bayesian GAMLSS via **R `bamlss`** through **`rpy2`** (requires R + `bamlss`). | Fitting is done in R (MCMC / inference as configured in the wrapper). | Python side obtains **posterior draws** of smooth effects; exported **μ** and **σ** samples per grid point (reshaping helpers align array orientation with the rest of the code). |

**Input scaling:** MC Dropout utilities include **`normalize_x` / `normalize_x_data`** so only **x** is standardized; **y** stays in original units (see comments in `MC_Dropout.py`).

### A.2 Loss functions (regression, PyTorch)

Defined next to the MC Dropout model in **`Models/MC_Dropout.py`**:

- **Gaussian NLL** for heteroscedastic regression (predicted variance per point).
- **β-NLL** — same residual term, weighted by a detached power of variance to mitigate variance underestimation (paper-style setup; default β often **0.5** in experiment kwargs).

Deep Ensemble training calls the **same** loss functions. BNN uses the **Pyro** likelihood (Gaussian with network output σ(x)), not these PyTorch losses. BAMLSS losses live inside the R fit.

### A.3 Variance decomposition (aleatoric vs epistemic)

After you have a stack of **M** (or **K**) predictive **means** `μ_m(x)` and **variances** `σ²_m(x)` on the grid:

- **Aleatoric (variance track):** \(\mathbb{E}_m[\sigma^2_m(x)]\) — mean of per-member variance predictions.
- **Epistemic (variance track):** \(\mathrm{Var}_m[\mu_m(x)]\) — variance of means across members.
- **Total (common decomposition):** aleatoric + epistemic.

This is implemented explicitly in **`mc_dropout_predict`** and **`ensemble_predict_deep`** in the `Models/*.py` files. The experiment code then passes the resulting curves (and/or raw `mu_samples`, `sigma2_samples`) to **`save_model_outputs`** and to metrics.

For **scalar** Gaussian NLL / CRPS on an aggregated predictive, **`utils/metrics.py`** defines **`compute_predictive_aggregation`**, which computes the mixture-style **μ\*** and **σ\*²** from the stacked samples (formula in the docstring). **`compute_gaussian_nll`**, **`compute_crps_gaussian`**, and **`compute_uncertainty_disentanglement`** (Spearman-style associations) build on those grids.

### A.4 Entropy decomposition (no extra training step)

**Entropy is not a separate training loss** in the default setup. Given the **same** `mu_samples` and `sigma2_samples` saved in npz, **`utils/entropy_uncertainty.py`** applies a **post-hoc** decomposition:

- Default **`entropy_method='moment_matched'`** → **`entropy_uncertainty_analytical_moment_matched`**: closed-form aleatoric / epistemic / total **differential entropy** consistent with the moment-matched total variance.
- Alternatives: **`analytical`** (older closed form), **`numerical`** (Monte Carlo mixture entropy; more costly).

Dispatch is centralized in **`entropy_uncertainty_by_method`**, which the experiment runners call after they already have `mu_samples` and `sigma2_samples` on the grid.

### A.5 End-to-end call chain (regression)

1. **Notebook** → `run_*_experiment` in **`utils/ood_experiments.py`** (or sample_size / noise_level / undersampling / ovb).
2. **Train / fit** → functions in **`Models/`** (or BAMLSS/R).
3. **Predictive sampling** → MC Dropout M passes, ensemble K passes, BNN/BAMLSS posterior draws → **`mu_samples`**, **`sigma2_samples`** shaped **(n_members, n_grid)** (with BAMLSS-specific orientation handled before save).
4. **Variance curves** → mean/var decomposition as above; optional **`compute_predictive_aggregation`** for μ\*, σ\*² and NLL/CRPS.
5. **Entropy curves** → **`entropy_uncertainty_by_method(..., entropy_method)`**.
6. **Persist / plot** → **`utils/results_save.py`**, **`utils/plotting.py`**.

So: **training optimizes NLL (or β-NLL) on μ and σ²** for torch models; **sampling** is either dropout randomness, ensemble diversity, or Bayesian posterior draws; **disentanglement on the variance side** is the standard mean-of-σ² / variance-of-μ split; **entropy** is a **downstream** transform of the saved sample tensors.

### A.6 Classification (separate stack)

**`utils/classification_models.py`** defines classification architectures (e.g. MC Dropout with **sampling softmax**), training loops, and predictors used by the classification notebooks. It does not use the regression modules in **`Models/`** for those experiments.
