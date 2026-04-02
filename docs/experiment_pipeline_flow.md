# Experiment pipeline: simulation → UQ models → decomposition → results

This document sketches the **end-to-end flow** implemented in the repository: synthetic data, training **uncertainty-quantifying (UQ)** predictors, **post-hoc decomposition** of total uncertainty into aleatoric vs epistemic components (“uncertainty disentanglement”, UD), and exported artifacts.

**Suggested format:** Markdown + **Mermaid** diagrams (render in GitHub/GitLab, VS Code “Markdown Preview”, Cursor, etc.). For a static PDF/slide, export the Mermaid blocks using [mermaid.live](https://mermaid.live) or a Pandoc/`mmdc` toolchain.

---

## High-level flow

```mermaid
flowchart TB
  subgraph SIM["1. Simulation (DGP)"]
    DGP["Synthetic data generator"]
    DGP --> TRAINXY["Training set (x, y)"]
    DGP --> GRID["Evaluation grid + clean f(x)"]
    DGP --> META["Knobs: τ, %, OOD range, regions, …"]
  end

  subgraph UQ["2. UQ predictors (training + forward passes)"]
    M["Model family"]
    M --> MC["MC Dropout"]
    M --> DE["Deep Ensemble"]
    M --> BNN["BNN (Pyro)"]
    M --> BAM["BAMLSS"]
    TRAINXY --> FIT["Fit on (x, y)"]
    FIT --> M
    GRID --> PRED["Predictive draws on grid"]
    M --> PRED
  end

  subgraph MEMBERS["Member-wise predictive law (per grid point)"]
    PRED --> REG["Regression: μ⁽ᵐ⁾(x), σ²⁽ᵐ⁾(x)  →  stacks μ_samples, sigma2_samples"]
    PRED --> CLF["Classification: IT logits→softmax p⁽ᵐ⁾  or  GL (μ, σ²) on logits"]
  end

  subgraph DECOMP["3. Uncertainty decomposition (UD)"]
    REG --> VREG["Variance: σ²_ale, σ²_epi, σ²_tot"]
    REG --> EREG["Entropy: analytical / numerical / k-NN / moment-matched (utils/entropy_uncertainty.py)"]
    CLF --> ITGL["IT: TU, AU, EU from {p⁽ᵐ⁾}  |  GL: AU, EU via MC softmax"]
  end

  subgraph OUT["4. Results & diagnostics"]
    VREG --> METR["MSE, NLL, CRPS, Spearman vs true noise / error"]
    EREG --> METR
    ITGL --> METR2["Accuracy, ECE, heatmaps, AU–EU correlation, …"]
    METR --> SAVE["Plots (results/.../plots)"]
    METR2 --> SAVE
    METR --> STATS["Tables (Excel/CSV, summaries)"]
    METR2 --> STATS
    MEMBERS --> NPZ["*raw_outputs*.npz (μ, σ² members + grid)"]
    NPZ --> BATCH["Optional: offline recomputation (e.g. scripts/recompute_entropy_*_from_npz.py)"]
    BATCH --> STATS
  end

  SIM --> UQ
```

---

## Regression (toy 1D): typical `utils/*_experiments.py` path

```mermaid
flowchart LR
  A["generate_toy_regression / OOD generator / undersampling helper"] --> B["(x_train, y_train), x_grid, y_clean"]
  B --> C["train_* + mc_dropout_predict / ensemble_predict / bnn_predict / bamlss_predict"]
  C --> D["mu_samples, sigma2_samples on grid"]
  D --> E["Aleatoric/epistemic/total variance (aggregated)"]
  D --> F["Entropy AU/EU/TU via entropy_uncertainty"]
  E --> G["Plots + compute_and_save_statistics_*"]
  F --> G
  D --> H["save_model_outputs → *raw_outputs*.npz"]
  H --> I["Batch scripts: moment-matched / numerical / kNN entropy"]
```

---

## Classification (2D blobs, etc.): `utils/classification_experiments.py`

```mermaid
flowchart LR
  A["simulate_dataset (or rotation/ring helpers)"] --> B["X_train, y_train, X_test, meta"]
  B --> C["train_mc_dropout_* / train_deep_ensemble_* / train_bnn_*"]
  C --> D["Member probabilities OR (μ, σ²) logits per draw"]
  D --> E["it_uncertainty / gl_uncertainty"]
  E --> F["AU, EU, TU (+ normalized variants)"]
  F --> G["Heatmaps, panels, save_statistics / Excel"]
```

---

## What lives where (quick map)

| Stage | Main locations |
|-------|----------------|
| Regression DGP | `utils/ood_experiments.py`, `sample_size_experiments.py`, `noise_level_experiments.py`, `undersampling_experiments.py`, `ovb_experiments.py`; notebooks under `Experiments/` |
| Classification DGP | `utils/classification_data.py`, `utils/classification_experiments.py` |
| Neural / BAMLSS training | `Models/`, `utils/classification_models.py` |
| Variance decomposition | Aggregates of `σ²` and `Var(μ)` over members (same experiment files) |
| Entropy (regression) | `utils/entropy_uncertainty.py`; batch recomputation under `scripts/recompute_entropy_*.py` |
| IT/GL decomposition (clf.) | `utils/classification_experiments.py` (`it_uncertainty`, `gl_uncertainty`) |
| Persistence | `utils/results_save.py` (`save_model_outputs`, `save_statistics`, …) |
| Default results tree | `results/<experiment>/plots`, `results/.../statistics`, `results/.../outputs` |

---

## Optional offline loop (entropy recomputation)

```mermaid
flowchart LR
  NPZ["*raw_outputs*.npz<br/>mu_samples, sigma2_samples, x_grid"] --> R["recompute_entropy_*_from_npz.py"]
  R --> NEW["New entropy curves / Excel / panels"]
```

OOD masks for batch tools are rebuilt from `x_grid` plus configured OOD intervals (see `utils/knn_entropy_regression.py`, `DEFAULT_OOD_RANGES`), not re-read from the `.npz` as a separate metadata field.

---

## Rendering tips

- **VS Code / Cursor:** open this file → Markdown preview.
- **GitHub:** commit and view the file; Mermaid renders automatically.
- **LaTeX thesis:** paste a PNG exported from mermaid.live, or use `\usepackage{mermaid}`-compatible packages if your template supports it.

If you want a single **PDF one-pager**, the first diagram is usually enough; keep the regression/classification subgraphs as appendices.
