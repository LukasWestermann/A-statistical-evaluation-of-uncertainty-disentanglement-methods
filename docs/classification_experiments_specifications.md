# Classification simulation study: DGP specifications (for thesis / Claude)

This file summarizes **how data are simulated** and the **fixed (non-swept) numerical defaults** used in the four classification experiment tracks, as implemented in `utils/classification_data.py` and the `Experiments/*.ipynb` notebooks.

**Note on “SED”:** In the codebase this is **RCD** (relative class distance), `rcd = d_between / σ_within`.

---

## 1. Which experiments share which simulator?

| Experiment | Notebook | Simulator | Same blob baseline as sample size? |
|------------|----------|-----------|-------------------------------------|
| **Sample size** | `Experiments/Classification_Sample_Size.ipynb` | `simulate_dataset` | — (reference baseline) |
| **Label noise** | `Experiments/Classification_Label_Noise.ipynb` | `simulate_dataset` | **Yes** (same `base_cfg` fields for data, except `eta` is swept) |
| **RCD** | `Experiments/Classification_RCD.ipynb` | `simulate_dataset` | **No** (different `N_train`, `blob_sigma`, and **boundary enrichment**; see §3) |
| **Ring OOD** | `Experiments/Classification_Ring_OOD.ipynb` | `simulate_ring_ood_dataset` | **No** (separate 2-class ring+gap geometry) |

---

## 2. Shared Gaussian-blob DGP (`simulate_dataset`) — mechanics

**Code:** `utils/classification_data.py`, function `simulate_dataset`.

### 2.1 Class geometry (3 classes)

- Template centers are a **unit equilateral triangle** in $\mathbb{R}^2$ (edge length 1), centered at the origin:
  - Top: $(0,\,1/\sqrt{3})$
  - Bottom-left: $(-0.5,\,-1/(2\sqrt{3}))$
  - Bottom-right: $(0.5,\,-1/(2\sqrt{3}))$
- **Scaled centers:**  
  $\mathbf{c}_k = \text{RCD} \cdot \sigma_{\text{blob}} \cdot \mathbf{c}_k^{(0)}$  
  so inter-center distances scale with **RCD** and within-class spread $\sigma_{\text{blob}}$.

### 2.2 Class-conditional inputs (Gaussian blobs)

- Class indices for training/test ID points are drawn with **uniform class priors** $1/3$ each (unless `class_priors` overridden).
- Given class $k$:  
  $\mathbf{X} \mid Y{=}k \sim \mathcal{N}(\mathbf{c}_k,\; \sigma_{\text{blob}}^2 \mathbf{I}_2)$  
  i.e. **isotropic Gaussian noise** with standard deviation $\sigma_{\text{blob}}$ per coordinate.

### 2.3 Score function and “soft” probabilities (for undersampling / OOD analysis)

- Squared Euclidean distances $d_k^2(\mathbf{x}) = \|\mathbf{x}-\mathbf{c}_k\|^2$.
- Scores (default biases $b_k=0$):  
  $s_k(\mathbf{x}) = -\dfrac{d_k^2(\mathbf{x})}{\tau} + b_k$.
- Softmax: $\pi_k(\mathbf{x}) \propto \exp(s_k(\mathbf{x}))$.

These $\pi_k$ are used for optional **undersampling** (boundary band / holes) and for **OOD** point analysis where applicable; **training labels** for the main blob path are the **hard** labels from the mixture sample unless modified below.

### 2.4 Label noise $\eta$ (`eta`)

- Applied **after** blob sampling (and after undersampling mask), **only to training labels**.
- For each training index, with probability $\eta$, the label is replaced by a **uniformly random other class** among the remaining $K-1$ classes (not a fixed pairwise flip).

### 2.5 Input noise (`sigma_in`, optional train/test split)

- Default: **no input noise** (`sigma_in = 0` in the notebooks below).
- If enabled: **additive** $\mathcal{N}(0,\sigma_{\text{in}}^2 \mathbf{I}_2)$ to coordinates (train and/or test via `sigma_in_train` / `sigma_in_test`).

### 2.6 Boundary enrichment (optional; **on in RCD notebook only**)

- If `boundary_enrichment.enabled`: extra points along **midlines between class centers**, with Gaussian jitter:
  - Along boundary direction: $\mathcal{N}(0,\,(\texttt{spread}\cdot\sigma_{\text{blob}})^2)$
  - Perpendicular: $\mathcal{N}(0,\,(\texttt{width}\cdot\sigma_{\text{blob}})^2)$
- Labels assigned by **nearest center** to those points.
- Appended to the training set before label noise / input noise.

### 2.7 Test set

- **ID test:** same blob mixture as training, size `N_test`, independent draw.
- **OOD test:** only if `ood_specs` / `N_ood_test` configured (not used in the four notebooks discussed here for the default cells).

---

## 3. Fixed numerical standards by notebook (data only)

### 3.1 Sample size + label noise (aligned blob baseline)

**Notebooks:** `Classification_Sample_Size.ipynb`, `Classification_Label_Noise.ipynb`  
**Shared data defaults:**

| Quantity | Value | Role |
|----------|-------|------|
| `num_classes` | 3 | |
| `N_train` | 1000 in `base_cfg` (placeholder); **sweep replaces** it per run with each value in `sample_sizes` (e.g. 100, 200, 500, 1000) | |
| `N_test` | 500 | |
| `rcd` | **3.0** | Sample-size notebook omits key → **same** default in code (`cfg.get("rcd", 3.0)`). Label-noise notebook sets **3.0** explicitly. |
| `blob_sigma` | **0.25** | Std dev of Gaussian blob per axis |
| `tau` | **0.2** | Softmax temperature in score $s_k$ |
| `sigma_in` | **0.0** | No additive Gaussian input noise |
| `eta` | **0.0** at baseline | **Swept** in label-noise runs only (`eta_values` e.g. `[0.0, 0.1, 0.3, 0.6]`; BNN subset may differ) |
| `seed` | 42 | |
| `boundary_enrichment` | **absent / disabled** | No extra boundary points |

**Implied within-class distribution:** $\mathcal{N}(\mathbf{c}_k,\,0.25^2 \mathbf{I}_2)$.

**What is swept (not “standard”):**

- **Sample size:** each run calls `simulate_dataset` with `N_train` ∈ `[100, 200, 500, 1000]` (BNN subset e.g. `[100, 1000]`); not subsampling from a fixed 1000 pool.
- **Label noise:** `eta` ∈ `[0.0, 0.1, 0.3, 0.6]` (BNN subset e.g. `[0.0, 0.6]`).

---

### 3.2 RCD experiments (different blob baseline in notebook)

**Notebook:** `Classification_RCD.ipynb`

| Quantity | Value | Notes |
|----------|-------|------|
| `N_train` | **500** | Not 1000 |
| `N_test` | 500 | |
| `num_classes` | 3 | |
| `blob_sigma` | **1.0** | Much wider blobs than sample-size / label-noise |
| `tau` | 0.2 | |
| `rcd` | **3.0** in `base_cfg` but **swept** in experiment (e.g. `[1, 7]` for MC/DE; BNN e.g. `[1.5, 3.0, 5.0]`) | |
| `sigma_in` | 0.0 | |
| `eta` | **0.0** (default; not set in `base_cfg`) | |
| `boundary_enrichment` | **enabled** | `n=50`, `spread=1.0`, `width=0.3` (in units of `blob_sigma` for spread/width as in code) |

So the **RCD** track is *not* identical to the sample-size/label-noise blob world unless you change the notebook to match §3.1.

---

## 4. Ring OOD (`simulate_ring_ood_dataset`) — separate DGP

**Notebook:** `Classification_Ring_OOD.ipynb`  
**Code:** `utils/classification_data.py` → `simulate_ring_ood_dataset`.

| Quantity | Default | Meaning |
|----------|---------|---------|
| Classes | **2** | Inner ring vs outer ring |
| `N_train` | 2000 | 1000 per class before shuffle |
| Inner radius range | **[0.5, 1.5]** | Class 0 |
| Outer radius range | **[2.5, 3.5]** | Class 1 |
| Gap (OOD eval only) | **[1.5, 2.5]** | `n_gap_points = 500`, **no training points** |
| Polar sampling | $\theta \sim \mathrm{Unif}(0,2\pi)$, $r \sim \mathrm{Unif}(r_{\min},r_{\max})$ | Then Cartesian; **not** area-uniform in $r$ |
| `coord_noise_std` | **0.1** | i.i.d. $\mathcal{N}(0, 0.1^2)$ on **both** coordinates, **training only** |
| `label_noise_rate` | **0.05** | Bernoulli flips **0↔1** on **training** labels |
| Gap points | No coordinate noise / no label noise in generator | Used for uncertainty only |
| `seed` | 42 | |

---

## 5. Model-side constants in notebooks (not DGP, but often cited next to simulation)

These are **training / inference** defaults in the notebook `base_cfg`, not part of `simulate_dataset` physics:

| Field | Sample size / Label noise | RCD notebook | Ring notebook |
|-------|---------------------------|--------------|---------------|
| `epochs` | 300 | 300 | 300 |
| `batch_size` | 32 | 32 | 32 |
| `lr` | 1e-3 | 1e-3 | 1e-3 |
| `dropout_p` | 0.25 | 0.25 | 0.25 |
| `mc_samples` | **50** | **100** | **100** |
| `gl_samples` | 100 | 100 | 100 |
| `K` (ensemble) | **5** | **20** | **20** |
| `hidden_width` | 32 | 32 | 32 |
| BNN `warmup` / `samples` / `chains` | 200 / 200 / 1 | 200 / 200 / 1 | (not in snippet; use ring notebook if extended) |

---

## 6. One-sentence summary for your chapter

- **Sample size** and **label noise** use the **same 3-class Gaussian blob simulator** with **RCD = 3**, **blob_sigma = 0.25**, **τ = 0.2**, **no input noise**, **no boundary enrichment**, **N_test = 500**, **seed 42**; they differ only in whether **N_train** or **η** is varied.  
- **RCD** uses the **same functional form** but the **published notebook** fixes **blob_sigma = 1**, **N_train = 500**, enables **boundary enrichment**, and varies **RCD**.  
- **Ring OOD** is a **binary ring + gap** layout with **Gaussian coordinate noise (0.1)** and **5% label flips** on training only, plus **500** gap test locations.

---

*Generated from repository sources: `utils/classification_data.py`, `Experiments/Classification_Sample_Size.ipynb`, `Experiments/Classification_Label_Noise.ipynb`, `Experiments/Classification_RCD.ipynb`, `Experiments/Classification_Ring_OOD.ipynb`.*
