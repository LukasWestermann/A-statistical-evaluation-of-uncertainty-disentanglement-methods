## Utility functions for saving results
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Module-level variables for directories (can be set by notebooks)
plots_dir = None
stats_dir = None

def _get_default_dirs():
    """Get default directories based on project structure"""
    # Try to find project root (look for results directory)
    current = Path.cwd()
    
    # If we're in Experiments folder, go up one level
    if current.name == 'Experiments':
        project_root = current.parent
    else:
        project_root = current
    
    # Look for results directory
    results_base = project_root / "results"
    if results_base.exists():
        # Try to find sample_size or use a default
        for subdir in results_base.iterdir():
            if subdir.is_dir():
                plots = subdir / "plots"
                stats = subdir / "statistics"
                if plots.exists() or stats.exists():
                    return plots if plots.exists() else subdir / "plots", \
                           stats if stats.exists() else subdir / "statistics"
        
        # Default to sample_size if it exists or create generic
        default_subdir = results_base / "sample_size"
        return default_subdir / "plots", default_subdir / "statistics"
    
    # Fallback: create in current directory
    return Path("results") / "plots", Path("results") / "statistics"

def sanitize_filename(name):
    """Convert a string to a valid filename"""
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    # Replace spaces with underscores and remove multiple underscores
    name = '_'.join(name.split())
    return name

def save_plot(fig, filename, subfolder=''):
    """Save a matplotlib figure to the results/plots directory"""
    global plots_dir
    if plots_dir is None:
        plots_dir, _ = _get_default_dirs()
    
    if subfolder:
        save_dir = plots_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = plots_dir
        save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / f"{sanitize_filename(filename)}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    return filepath

def save_statistics(data_dict, filename, subfolder='', save_excel=True):
    """Save statistics dictionary to CSV and optionally Excel"""
    global stats_dir
    if stats_dir is None:
        _, stats_dir = _get_default_dirs()
    
    if subfolder:
        save_dir = stats_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = stats_dir
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame if it's a dict
    if isinstance(data_dict, dict):
        df = pd.DataFrame(data_dict)
    else:
        df = data_dict
    
    # Save as CSV
    csv_filepath = save_dir / f"{sanitize_filename(filename)}.csv"
    df.to_csv(csv_filepath, index=False)
    print(f"Saved statistics (CSV): {csv_filepath}")
    
    # Save as Excel if requested
    if save_excel:
        try:
            excel_filepath = save_dir / f"{sanitize_filename(filename)}.xlsx"
            df.to_excel(excel_filepath, index=False, engine='openpyxl')
            print(f"Saved statistics (Excel): {excel_filepath}")
            return csv_filepath, excel_filepath
        except ImportError:
            print("Warning: openpyxl not available. Excel file not saved. Only CSV saved.")
            print("Install openpyxl with: pip install openpyxl")
            return csv_filepath
    
    return csv_filepath

def save_summary_text(text, filename, subfolder=''):
    """Save summary text to a file"""
    global stats_dir
    if stats_dir is None:
        _, stats_dir = _get_default_dirs()
    
    if subfolder:
        save_dir = stats_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = stats_dir
        save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / f"{sanitize_filename(filename)}.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved summary: {filepath}")
    return filepath

def save_summary_statistics(percentages, avg_ale_norm_list, avg_epi_norm_list, 
                           avg_tot_norm_list, correlation_list, function_name, 
                           noise_type='heteroscedastic', func_type='', model_name='',
                           mse_list=None):
    """Helper function to save summary statistics and create summary plot
    
    Args:
        percentages: List of training data percentages
        avg_ale_norm_list: List of normalized average aleatoric uncertainties
        avg_epi_norm_list: List of normalized average epistemic uncertainties
        avg_tot_norm_list: List of normalized average total uncertainties
        correlation_list: List of correlations between epistemic and aleatoric uncertainties
        function_name: Name of the function (e.g., 'Linear', 'Sinusoidal')
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        model_name: Name of the model (e.g., 'MC_Dropout', 'Deep_Ensemble') - optional
        mse_list: Optional list of MSE values to include in statistics and plots
    
    Returns:
        tuple: (stats_df, fig) - DataFrame with statistics and matplotlib figure
    """
    # Create DataFrame with summary statistics
    stats_dict = {
        'Percentage': percentages,
        'Avg_Aleatoric_norm': avg_ale_norm_list,
        'Avg_Epistemic_norm': avg_epi_norm_list,
        'Avg_Total_norm': avg_tot_norm_list,
        'Correlation_Epi_Ale': correlation_list
    }
    
    # Add MSE if provided
    if mse_list is not None:
        stats_dict['MSE'] = mse_list
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Build filename with optional model prefix
    base_filename = f"uncertainties_summary_{function_name}_{noise_type}"
    if model_name:
        filename = f"{model_name}_{base_filename}"
    else:
        filename = base_filename
    
    # Save statistics to CSV and Excel (Excel is saved automatically)
    save_statistics(stats_df, filename, 
                    subfolder=f"{noise_type}/{func_type}", save_excel=True)
    
    # Create and save summary plots
    # Use 3 subplots if MSE is provided, otherwise 2
    if mse_list is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Add model name to title if provided
    title_suffix = f" - {model_name}" if model_name else ""
    
    # Plot 1: Normalized Average Uncertainties
    ax1.plot(percentages, avg_ale_norm_list, 'o-', linewidth=2, markersize=8, 
             label='Aleatoric Uncertainty', color='green')
    ax1.plot(percentages, avg_epi_norm_list, 's-', linewidth=2, markersize=8, 
             label='Epistemic Uncertainty', color='orange')
    ax1.plot(percentages, avg_tot_norm_list, '^-', linewidth=2, markersize=8, 
             label='Total Uncertainty', color='blue')
    ax1.set_xlabel('Training Data Percentage (%)', fontsize=12)
    ax1.set_ylabel('Normalized Average Uncertainty', fontsize=12)
    ax1.set_title(f'Normalized Average Uncertainties vs Training Data Percentage\n{function_name} Function ({noise_type.capitalize()}){title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Correlation between Epistemic and Aleatoric Uncertainties
    ax2.plot(percentages, correlation_list, 'D-', linewidth=2, markersize=8, 
             label='Correlation (Epi-Ale)', color='purple')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Training Data Percentage (%)', fontsize=12)
    ax2.set_ylabel('Correlation Coefficient', fontsize=12)
    ax2.set_title(f'Correlation: Epistemic vs Aleatoric Uncertainty\n{function_name} Function ({noise_type.capitalize()}){title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)
    ax2.set_ylim(-1.05, 1.05)
    
    # Plot 3: MSE (if provided)
    if mse_list is not None:
        ax3.plot(percentages, mse_list, '*-', linewidth=2, markersize=8, 
                 label='MSE', color='red')
        ax3.set_xlabel('Training Data Percentage (%)', fontsize=12)
        ax3.set_ylabel('Mean Squared Error', fontsize=12)
        ax3.set_title(f'MSE vs Training Data Percentage\n{function_name} Function ({noise_type.capitalize()}){title_suffix}', 
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 105)
        ax3.set_yscale('log')  # Use log scale for MSE as it can vary widely
    
    plt.tight_layout()
    
    save_plot(fig, filename, 
              subfolder=f"{noise_type}/{func_type}")
    
    return stats_df, fig

def save_summary_statistics_noise_level(tau_values, avg_ale_norm_list, avg_epi_norm_list, 
                                       avg_tot_norm_list, correlation_list, mse_list,
                                       function_name, distribution='normal',
                                       noise_type='heteroscedastic', func_type='', model_name=''):
    """Helper function to save summary statistics and create summary plot for noise level experiments
    
    Args:
        tau_values: List of tau (noise level) values
        avg_ale_norm_list: List of normalized average aleatoric uncertainties
        avg_epi_norm_list: List of normalized average epistemic uncertainties
        avg_tot_norm_list: List of normalized average total uncertainties
        correlation_list: List of correlations between epistemic and aleatoric uncertainties
        mse_list: List of MSE values
        function_name: Name of the function (e.g., 'Linear', 'Sinusoidal')
        distribution: Noise distribution ('normal' or 'laplace')
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        model_name: Name of the model (e.g., 'MC_Dropout', 'Deep_Ensemble') - optional
    
    Returns:
        tuple: (stats_df, fig) - DataFrame with statistics and matplotlib figure
    """
    # Create DataFrame with summary statistics
    stats_df = pd.DataFrame({
        'Tau': tau_values,
        'Distribution': [distribution] * len(tau_values),
        'Avg_Aleatoric_norm': avg_ale_norm_list,
        'Avg_Epistemic_norm': avg_epi_norm_list,
        'Avg_Total_norm': avg_tot_norm_list,
        'Correlation_Epi_Ale': correlation_list,
        'MSE': mse_list
    })
    
    # Build filename with optional model prefix and distribution
    base_filename = f"uncertainties_summary_{function_name}_{noise_type}_{distribution}"
    if model_name:
        filename = f"{model_name}_{base_filename}"
    else:
        filename = base_filename
    
    # Save statistics to CSV and Excel (Excel is saved automatically)
    save_statistics(stats_df, filename, 
                    subfolder=f"{noise_type}/{func_type}/{distribution}", save_excel=True)
    
    # Create and save summary plots (uncertainties, correlations, and MSE)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # Add model name to title if provided
    title_suffix = f" - {model_name}" if model_name else ""
    
    # Plot 1: Normalized Average Uncertainties
    ax1.plot(tau_values, avg_ale_norm_list, 'o-', linewidth=2, markersize=8, 
             label='Aleatoric Uncertainty', color='green')
    ax1.plot(tau_values, avg_epi_norm_list, 's-', linewidth=2, markersize=8, 
             label='Epistemic Uncertainty', color='orange')
    ax1.plot(tau_values, avg_tot_norm_list, '^-', linewidth=2, markersize=8, 
             label='Total Uncertainty', color='blue')
    ax1.set_xlabel('Tau (Noise Level)', fontsize=12)
    ax1.set_ylabel('Normalized Average Uncertainty', fontsize=12)
    ax1.set_title(f'Normalized Average Uncertainties vs Tau\n{function_name} Function ({noise_type.capitalize()}, {distribution}){title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation between Epistemic and Aleatoric Uncertainties
    ax2.plot(tau_values, correlation_list, 'D-', linewidth=2, markersize=8, 
             label='Correlation (Epi-Ale)', color='purple')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Tau (Noise Level)', fontsize=12)
    ax2.set_ylabel('Correlation Coefficient', fontsize=12)
    ax2.set_title(f'Correlation: Epistemic vs Aleatoric Uncertainty\n{function_name} Function ({noise_type.capitalize()}, {distribution}){title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.05, 1.05)
    
    # Plot 3: MSE
    ax3.plot(tau_values, mse_list, '*-', linewidth=2, markersize=8, 
             label='MSE', color='red')
    ax3.set_xlabel('Tau (Noise Level)', fontsize=12)
    ax3.set_ylabel('Mean Squared Error', fontsize=12)
    ax3.set_title(f'MSE vs Tau\n{function_name} Function ({noise_type.capitalize()}, {distribution}){title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Use log scale for MSE as it can vary widely
    
    plt.tight_layout()
    
    save_plot(fig, filename, 
              subfolder=f"{noise_type}/{func_type}/{distribution}")
    
    return stats_df, fig