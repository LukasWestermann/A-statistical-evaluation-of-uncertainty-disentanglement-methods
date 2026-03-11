"""Print per-point uncertainty estimates for BAMLSS OOD (x <= 12.5). Also saves as a dataframe (CSV)."""
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
npz_path = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "linear" / "20260107_BAMLSS_raw_outputs.npz"
X_MAX = 12.5

data = np.load(npz_path, allow_pickle=True)
mu_samples = data["mu_samples"]
sigma2_samples = data["sigma2_samples"]
x_grid = data["x_grid"]
n_grid = len(x_grid.ravel())
if mu_samples.shape[0] == n_grid and mu_samples.shape[1] != n_grid:
    mu_samples = mu_samples.T
    sigma2_samples = sigma2_samples.T

mu_pred = np.mean(mu_samples, axis=0)
ale_var = np.mean(sigma2_samples, axis=0)
epi_var = np.var(mu_samples, axis=0)
tot_var = ale_var + epi_var
x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()

crop = x <= X_MAX
x_c = x[crop]
mu_pred_c = mu_pred[crop]
ale_var_c = ale_var[crop]
epi_var_c = epi_var[crop]
tot_var_c = tot_var[crop]
ale_std_c = np.sqrt(ale_var_c)
epi_std_c = np.sqrt(epi_var_c)
tot_std_c = np.sqrt(tot_var_c)

print("Per-point uncertainty estimates (x <= 12.5), first 20 and last 5 points:")
print("BAMLSS OOD - heteroscedastic / linear")
header = f"{'x':>8} {'mu_pred':>10} {'ale_var':>10} {'epi_var':>10} {'tot_var':>10} {'ale_std':>10} {'epi_std':>10} {'tot_std':>10}"
print(header)
print("-" * 88)
for i in range(min(20, len(x_c))):
    print(f"{x_c[i]:8.4f} {mu_pred_c[i]:10.6f} {ale_var_c[i]:10.6f} {epi_var_c[i]:10.6f} {tot_var_c[i]:10.6f} {ale_std_c[i]:10.6f} {epi_std_c[i]:10.6f} {tot_std_c[i]:10.6f}")
if len(x_c) > 25:
    print("...")
    for i in range(max(0, len(x_c) - 5), len(x_c)):
        print(f"{x_c[i]:8.4f} {mu_pred_c[i]:10.6f} {ale_var_c[i]:10.6f} {epi_var_c[i]:10.6f} {tot_var_c[i]:10.6f} {ale_std_c[i]:10.6f} {epi_std_c[i]:10.6f} {tot_std_c[i]:10.6f}")
print(f"\nTotal points (x <= 12.5): {len(x_c)}")

# Save as dataframe (CSV)
df = pd.DataFrame({
    "x": x_c,
    "mu_pred": mu_pred_c,
    "ale_var": ale_var_c,
    "epi_var": epi_var_c,
    "tot_var": tot_var_c,
    "ale_std": ale_std_c,
    "epi_std": epi_std_c,
    "tot_std": tot_std_c,
})
out_dir = project_root / "results" / "ood" / "statistics" / "ood" / "heteroscedastic" / "linear"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "BAMLSS_OOD_uncertainties_per_point_x12.5.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved dataframe to {out_path}")
print(df)