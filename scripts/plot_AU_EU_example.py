import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

# --- Domain & clean curve ---
x_min, x_max = 0.0, 13.0
id_lo, id_hi = 3.0, 10.0

x_line = np.linspace(x_min, x_max, 600)
y_line = 10.0 * np.sin(x_line)

# --- Noisy in-distribution samples only on [id_lo, id_hi] ---
rng = np.random.default_rng(0)
n_train = 120
x_train = rng.uniform(id_lo, id_hi, size=n_train)
noise_std = 5
y_train = 10.0 * np.sin(x_train) + rng.normal(0.0, noise_std, size=n_train)

# --- Figure ---
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
ax.set_facecolor("#e8eef2")
ax.grid(True, color="white", linewidth=1.0, alpha=0.9)

ax.plot(x_line, y_line, color="#2c2c5c", linewidth=2.2, zorder=2, label="True $f(x)$")
ax.scatter(
    x_train, y_train,
    s=38, c="#7b1020", alpha=0.85, edgecolors="#4a0a14", linewidths=0.4, zorder=3,
    label="Training data",
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(-18, 18)
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$y$", fontsize=12)

# --- Aleatoric: vertical double arrow at left of ID band (noise spread) ---
xa = id_lo
y_spread_lo = -13.0   # bottom of double arrow
y_spread_hi = 6
ax.annotate(
    "",
    xy=(xa, y_spread_hi),
    xytext=(xa, y_spread_lo),
    arrowprops=dict(arrowstyle="<->", color="crimson", lw=2.0, shrinkA=0, shrinkB=0),
    zorder=4,
)
txt_ale = ax.text(
    xa + 0.35, 0.5 * (y_spread_lo + y_spread_hi),
    "Aleatoric",
    color="crimson",
    fontsize=13,
    fontweight="bold",
    fontfamily="serif",
    va="center",
    ha="left",
    zorder=5,
)
txt_ale.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

# --- Epistemic: horizontal double arrow on empty OOD tail [id_hi, x_max] ---
y_epi = -11.0
ax.annotate(
    "",
    xy=(x_max - 0.15, y_epi),
    xytext=(id_hi + 0.15, y_epi),
    arrowprops=dict(arrowstyle="<->", color="darkorange", lw=2.0, shrinkA=0, shrinkB=0),
    zorder=4,
)
txt_epi = ax.text(
    0.5 * (id_hi + x_max), y_epi - 1.6,
    "Epistemic",
    color="darkorange",
    fontsize=13,
    fontweight="bold",
    fontfamily="serif",
    ha="center",
    va="top",
    zorder=5,
)
txt_epi.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

ax.legend(loc="upper right", framealpha=0.92)
plt.tight_layout()
plt.show()