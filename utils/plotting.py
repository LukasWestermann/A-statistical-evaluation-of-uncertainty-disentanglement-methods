import matplotlib.pyplot as plt
import numpy as np
from utils.results_save import save_plot


def plot_toy_data(x_train, y_train, x_grid, y_clean, title="Toy Regression Data", save_plot_file=True):
    """Plot the training data and clean function"""
    fig = plt.figure(figsize=(12, 6))
    
    # Plot training data points
    plt.scatter(x_train, y_train, alpha=0.6, s=20, label="Training data", color='blue')
    
    # Plot clean function
    plt.plot(x_grid, y_clean, 'r--', linewidth=2, label="Clean f(x) = 0.7x + 0.5")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot_file:
        save_plot(fig, title, subfolder='toy_data')
    
    plt.show()
    plt.close(fig)



# Simplified plotting function without OOD
def plot_uncertainties_no_ood(x_train_subset, y_train_subset, x_grid, y_clean, mu_pred, ale_var, epi_var, tot_var, title, noise_type='heteroscedastic', func_type=''):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    x = x_grid[:, 0]
    
    # Ensure mu_pred is 1D for plotting (handle both 1D and 2D inputs)
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    
    # Plot 1: Predictive mean + Total uncertainty
    axes[0].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.6, s=20, color='blue', label="Training data", zorder=3)
    axes[0].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[0].fill_between(x, mu_pred - np.sqrt(tot_var), mu_pred + np.sqrt(tot_var), 
                        alpha=0.3, color='blue', label="±σ(total)")
    axes[0].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean f(x) = 0.7x + 0.5")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title}: Predictive Mean + Total Uncertainty")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Aleatoric uncertainty only
    axes[1].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.6, s=20, color='blue', label="Training data", zorder=3)
    axes[1].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[1].fill_between(x, mu_pred - np.sqrt(ale_var), mu_pred + np.sqrt(ale_var), 
                        alpha=0.3, color='green', label="±σ(aleatoric)")
    axes[1].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean f(x) = 0.7x + 0.5")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title}: Predictive Mean + Aleatoric Uncertainty")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Predictive mean + Epistemic uncertainty only
    axes[2].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.6, s=20, color='blue', label="Training data", zorder=3)
    axes[2].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[2].fill_between(x, mu_pred - np.sqrt(epi_var), mu_pred + np.sqrt(epi_var), 
                        alpha=0.3, color='orange', label="±σ(epistemic)")
    axes[2].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean f(x) = 0.7x + 0.5")
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title}: Predictive Mean + Epistemic Uncertainty")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot with organized folder structure: uncertainties/{noise_type}/{func_type}/
    subfolder = f"uncertainties/{noise_type}/{func_type}" if func_type else f"uncertainties/{noise_type}"
    save_plot(fig, title, subfolder=subfolder)
    
    plt.show()
    plt.close(fig)
