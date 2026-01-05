## Utility functions for saving results
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from datetime import datetime

# Module-level variables for directories (can be set by notebooks)
plots_dir = None
stats_dir = None
outputs_dir = None

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
    
    # Save as Excel if requested
    if save_excel:
        try:
            excel_filepath = save_dir / f"{sanitize_filename(filename)}.xlsx"
            df.to_excel(excel_filepath, index=False, engine='openpyxl')
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
                           mse_list=None, date=None, dropout_p=None, mc_samples=None, n_nets=None,
                           nll_list=None, crps_list=None,
                           spearman_aleatoric_list=None, spearman_epistemic_list=None):
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
        date: Optional date string in YYYYMMDD format
        dropout_p: Optional dropout probability for MC Dropout (float)
        mc_samples: Optional number of MC samples for MC Dropout (int)
        n_nets: Optional number of nets for Deep Ensemble (int)
    
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
    
    # Always add new metrics columns (use provided values or None)
    stats_dict['NLL'] = nll_list if nll_list is not None else [None] * len(percentages)
    stats_dict['CRPS'] = crps_list if crps_list is not None else [None] * len(percentages)
    stats_dict['Spearman_Aleatoric'] = spearman_aleatoric_list if spearman_aleatoric_list is not None else [None] * len(percentages)
    stats_dict['Spearman_Epistemic'] = spearman_epistemic_list if spearman_epistemic_list is not None else [None] * len(percentages)
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Build filename with optional date and model parameters
    base_filename = f"uncertainties_summary_{function_name}_{noise_type}"
    
    # Build parameter strings
    param_parts = []
    if model_name:
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                # Format dropout probability as p0.2 or p0.25 (keep decimal point)
                param_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                param_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                param_parts.append(f"K{n_nets}")
        
        # Combine model name and parameters
        if param_parts:
            model_prefix = f"{model_name}_{'_'.join(param_parts)}"
        else:
            model_prefix = model_name
        
        filename = f"{model_prefix}_{base_filename}"
    else:
        filename = base_filename
    
    # Add date prefix if provided
    if date:
        filename = f"{date}_{filename}"
    
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
                                       noise_type='heteroscedastic', func_type='', model_name='',
                                       date=None, dropout_p=None, mc_samples=None, n_nets=None,
                                       nll_list=None, crps_list=None,
                                       spearman_aleatoric_list=None, spearman_epistemic_list=None):
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
        date: Optional date string in YYYYMMDD format
        dropout_p: Optional dropout probability for MC Dropout (float)
        mc_samples: Optional number of MC samples for MC Dropout (int)
        n_nets: Optional number of nets for Deep Ensemble (int)
    
    Returns:
        tuple: (stats_df, fig) - DataFrame with statistics and matplotlib figure
    """
    # Create DataFrame with summary statistics
    stats_dict = {
        'Tau': tau_values,
        'Distribution': [distribution] * len(tau_values),
        'Avg_Aleatoric_norm': avg_ale_norm_list,
        'Avg_Epistemic_norm': avg_epi_norm_list,
        'Avg_Total_norm': avg_tot_norm_list,
        'Correlation_Epi_Ale': correlation_list,
        'MSE': mse_list
    }
    
    # Always add new metrics columns (use provided values or None)
    stats_dict['NLL'] = nll_list if nll_list is not None else [None] * len(tau_values)
    stats_dict['CRPS'] = crps_list if crps_list is not None else [None] * len(tau_values)
    stats_dict['Spearman_Aleatoric'] = spearman_aleatoric_list if spearman_aleatoric_list is not None else [None] * len(tau_values)
    stats_dict['Spearman_Epistemic'] = spearman_epistemic_list if spearman_epistemic_list is not None else [None] * len(tau_values)
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Build filename with optional date and model parameters
    base_filename = f"uncertainties_summary_{function_name}_{noise_type}_{distribution}"
    
    # Build parameter strings
    param_parts = []
    if model_name:
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                # Format dropout probability as p0.2 or p0.25 (keep decimal point)
                param_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                param_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                param_parts.append(f"K{n_nets}")
        
        # Combine model name and parameters
        if param_parts:
            model_prefix = f"{model_name}_{'_'.join(param_parts)}"
        else:
            model_prefix = model_name
        
        filename = f"{model_prefix}_{base_filename}"
    else:
        filename = base_filename
    
    # Add date prefix if provided
    if date:
        filename = f"{date}_{filename}"
    
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

def save_summary_statistics_ood(avg_ale_norm_list, avg_epi_norm_list, 
                                avg_tot_norm_list, correlation_list, mse_list,
                                function_name, noise_type='heteroscedastic', 
                                func_type='', model_name='', region_type='ID',
                                date=None, dropout_p=None, mc_samples=None, n_nets=None,
                                nll_list=None, crps_list=None,
                                spearman_aleatoric_list=None, spearman_epistemic_list=None):
    """Helper function to save summary statistics for OOD experiments
    
    Args:
        avg_ale_norm_list: List of normalized average aleatoric uncertainties
        avg_epi_norm_list: List of normalized average epistemic uncertainties
        avg_tot_norm_list: List of normalized average total uncertainties
        correlation_list: List of correlations between epistemic and aleatoric uncertainties
        mse_list: List of MSE values
        function_name: Name of the function (e.g., 'Linear', 'Sinusoidal')
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        model_name: Name of the model (e.g., 'MC_Dropout', 'Deep_Ensemble') - optional
        region_type: Region type ('ID', 'OOD', or 'Combined')
        date: Optional date string in YYYYMMDD format
        dropout_p: Optional dropout probability for MC Dropout (float)
        mc_samples: Optional number of MC samples for MC Dropout (int)
        n_nets: Optional number of nets for Deep Ensemble (int)
        nll_list: Optional list of NLL values
        crps_list: Optional list of CRPS values
        spearman_aleatoric_list: Optional list of Spearman correlations (aleatoric)
        spearman_epistemic_list: Optional list of Spearman correlations (epistemic)
    
    Returns:
        tuple: (stats_df, None) - DataFrame with statistics (plots removed, only uncertainty plots with data points are displayed)
    """
    # Create DataFrame with summary statistics
    stats_dict = {
        'Avg_Aleatoric_Variance': avg_ale_norm_list,
        'Avg_Epistemic_Variance': avg_epi_norm_list,
        'Avg_Total_Variance': avg_tot_norm_list,
        'Correlation_Epi_Ale': correlation_list,
        'MSE': mse_list
    }
    
    # Always add new metrics columns (use provided values or None)
    stats_dict['NLL'] = nll_list if nll_list is not None else [None] * len(avg_ale_norm_list)
    stats_dict['CRPS'] = crps_list if crps_list is not None else [None] * len(avg_ale_norm_list)
    stats_dict['Spearman_Aleatoric'] = spearman_aleatoric_list if spearman_aleatoric_list is not None else [None] * len(avg_ale_norm_list)
    stats_dict['Spearman_Epistemic'] = spearman_epistemic_list if spearman_epistemic_list is not None else [None] * len(avg_ale_norm_list)
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Build filename with optional date and model parameters
    base_filename = f"{region_type}_uncertainties_summary_{function_name}_{noise_type}_variance"
    
    # Build parameter strings
    param_parts = []
    if model_name:
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                param_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                param_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                param_parts.append(f"K{n_nets}")
        
        # Combine model name and parameters
        if param_parts:
            model_prefix = f"{model_name}_{'_'.join(param_parts)}"
        else:
            model_prefix = model_name
        
        filename = f"{model_prefix}_{base_filename}"
    else:
        filename = base_filename
    
    # Add date prefix if provided
    if date:
        filename = f"{date}_{filename}"
    
    # Save statistics to CSV and Excel (Excel is saved automatically)
    save_statistics(stats_df, filename, 
                    subfolder=f"ood/{noise_type}/{func_type}", save_excel=True)
    
    # Summary plots removed - only uncertainty plots with data points are displayed
    # All statistics are printed to console and saved to Excel/CSV
    
    return stats_df, None

def save_summary_statistics_undersampling(avg_ale_norm_list, avg_epi_norm_list, 
                                         avg_tot_norm_list, correlation_list, mse_list,
                                         function_name, noise_type='heteroscedastic', 
                                         func_type='', model_name='', region_name='Region',
                                         date=None, dropout_p=None, mc_samples=None, n_nets=None,
                                         density_factor=None,
                                         nll_list=None, crps_list=None,
                                         spearman_aleatoric_list=None, spearman_epistemic_list=None):
    """Helper function to save summary statistics and create summary plot for undersampling experiments
    
    Args:
        avg_ale_norm_list: List of normalized average aleatoric uncertainties
        avg_epi_norm_list: List of normalized average epistemic uncertainties
        avg_tot_norm_list: List of normalized average total uncertainties
        correlation_list: List of correlations between epistemic and aleatoric uncertainties
        mse_list: List of MSE values
        function_name: Name of the function (e.g., 'Linear', 'Sinusoidal')
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        model_name: Name of the model (e.g., 'MC_Dropout', 'Deep_Ensemble') - optional
        region_name: Region name (e.g., 'Region_1', 'Undersampled', 'Well_sampled')
        date: Optional date string in YYYYMMDD format
        dropout_p: Optional dropout probability for MC Dropout (float)
        mc_samples: Optional number of MC samples for MC Dropout (int)
        n_nets: Optional number of nets for Deep Ensemble (int)
        density_factor: Optional density factor for filename
    
    Returns:
        tuple: (stats_df, fig) - DataFrame with statistics and matplotlib figure
    """
    # Create DataFrame with summary statistics
    stats_dict = {
        'Avg_Aleatoric_norm': avg_ale_norm_list,
        'Avg_Epistemic_norm': avg_epi_norm_list,
        'Avg_Total_norm': avg_tot_norm_list,
        'Correlation_Epi_Ale': correlation_list,
        'MSE': mse_list
    }
    
    # Always add new metrics columns (use provided values or None)
    stats_dict['NLL'] = nll_list if nll_list is not None else [None] * len(avg_ale_norm_list)
    stats_dict['CRPS'] = crps_list if crps_list is not None else [None] * len(avg_ale_norm_list)
    stats_dict['Spearman_Aleatoric'] = spearman_aleatoric_list if spearman_aleatoric_list is not None else [None] * len(avg_ale_norm_list)
    stats_dict['Spearman_Epistemic'] = spearman_epistemic_list if spearman_epistemic_list is not None else [None] * len(avg_ale_norm_list)
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Build filename with optional date and model parameters
    density_str = f"_density{density_factor}" if density_factor is not None else ""
    base_filename = f"{region_name}{density_str}_uncertainties_summary_{function_name}_{noise_type}"
    
    # Build parameter strings
    param_parts = []
    if model_name:
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                param_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                param_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                param_parts.append(f"K{n_nets}")
        
        # Combine model name and parameters
        if param_parts:
            model_prefix = f"{model_name}_{'_'.join(param_parts)}"
        else:
            model_prefix = model_name
        
        filename = f"{model_prefix}_{base_filename}"
    else:
        filename = base_filename
    
    # Add date prefix if provided
    if date:
        filename = f"{date}_{filename}"
    
    # Save statistics to CSV and Excel (Excel is saved automatically)
    save_statistics(stats_df, filename, 
                    subfolder=f"undersampling/{noise_type}/{func_type}", save_excel=True)
    
    # Skip plot creation - only return statistics DataFrame
    return stats_df, None


# ========== Entropy-Based Saving Functions ==========

def save_summary_statistics_entropy(percentages, avg_ale_entropy_list, avg_epi_entropy_list, 
                                   avg_tot_entropy_list, correlation_list, function_name, 
                                   noise_type='heteroscedastic', func_type='', model_name='',
                                   mse_list=None, date=None, dropout_p=None, mc_samples=None, n_nets=None):
    """Helper function to save normalized entropy-based summary statistics and create summary plot
    
    Args:
        percentages: List of training data percentages
        avg_ale_entropy_list: List of normalized average aleatoric entropies (in [0, 1])
        avg_epi_entropy_list: List of normalized average epistemic entropies (in [0, 1])
        avg_tot_entropy_list: List of normalized average total entropies (in [0, 1])
        correlation_list: List of correlations between epistemic and aleatoric uncertainties
        function_name: Name of the function (e.g., 'Linear', 'Sinusoidal')
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        model_name: Name of the model (e.g., 'MC_Dropout', 'Deep_Ensemble') - optional
        mse_list: Optional list of MSE values to include in statistics and plots
        date: Optional date string in YYYYMMDD format
        dropout_p: Optional dropout probability for MC Dropout (float)
        mc_samples: Optional number of MC samples for MC Dropout (int)
        n_nets: Optional number of nets for Deep Ensemble (int)
    
    Returns:
        tuple: (stats_df, fig) - DataFrame with statistics and matplotlib figure
    """
    # Create DataFrame with summary statistics
    stats_dict = {
        'Percentage': percentages,
        'Avg_Aleatoric_Entropy_norm': avg_ale_entropy_list,
        'Avg_Epistemic_Entropy_norm': avg_epi_entropy_list,
        'Avg_Total_Entropy_norm': avg_tot_entropy_list,
        'Correlation_Epi_Ale': correlation_list
    }
    
    # Add MSE if provided
    if mse_list is not None:
        stats_dict['MSE'] = mse_list
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Build filename with optional date and model parameters
    base_filename = f"uncertainties_summary_{function_name}_{noise_type}_entropy"
    
    # Build parameter strings
    param_parts = []
    if model_name:
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                param_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                param_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                param_parts.append(f"K{n_nets}")
        
        if param_parts:
            model_prefix = f"{model_name}_{'_'.join(param_parts)}"
        else:
            model_prefix = model_name
        
        filename = f"{model_prefix}_{base_filename}"
    else:
        filename = base_filename
    
    # Add date prefix if provided
    if date:
        filename = f"{date}_{filename}"
    
    # Save statistics to CSV and Excel
    save_statistics(stats_df, filename, 
                    subfolder=f"{noise_type}/{func_type}", save_excel=True)
    
    # Create and save summary plots
    if mse_list is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    title_suffix = f" - {model_name}" if model_name else ""
    
    # Plot 1: Average Entropies
    ax1.plot(percentages, avg_ale_entropy_list, 'o-', linewidth=2, markersize=8, 
             label='Aleatoric Entropy', color='green')
    ax1.plot(percentages, avg_epi_entropy_list, 's-', linewidth=2, markersize=8, 
             label='Epistemic Entropy', color='orange')
    ax1.plot(percentages, avg_tot_entropy_list, '^-', linewidth=2, markersize=8, 
             label='Total Entropy', color='blue')
    ax1.set_xlabel('Training Data Percentage (%)', fontsize=12)
    ax1.set_ylabel('Normalized Average Entropy', fontsize=12)
    ax1.set_title(f'Normalized Average Entropies vs Training Data Percentage\n{function_name} Function ({noise_type.capitalize()}){title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)
    
    # Plot 2: Correlation
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
        ax3.set_yscale('log')
    
    plt.tight_layout()
    
    save_plot(fig, filename, 
              subfolder=f"{noise_type}/{func_type}")
    
    return stats_df, fig


def save_summary_statistics_entropy_noise_level(tau_values, avg_ale_entropy_list, avg_epi_entropy_list, 
                                               avg_tot_entropy_list, correlation_list, mse_list,
                                               function_name, distribution='normal',
                                               noise_type='heteroscedastic', func_type='', model_name='',
                                               date=None, dropout_p=None, mc_samples=None, n_nets=None):
    """Helper function to save normalized entropy-based summary statistics for noise level experiments
    
    Args:
        tau_values: List of tau (noise level) values
        avg_ale_entropy_list: List of normalized average aleatoric entropies (in [0, 1])
        avg_epi_entropy_list: List of normalized average epistemic entropies (in [0, 1])
        avg_tot_entropy_list: List of normalized average total entropies (in [0, 1])
        correlation_list: List of correlations between epistemic and aleatoric uncertainties
        mse_list: List of MSE values
        function_name: Name of the function (e.g., 'Linear', 'Sinusoidal')
        distribution: Noise distribution ('normal' or 'laplace')
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        model_name: Name of the model (e.g., 'MC_Dropout', 'Deep_Ensemble') - optional
        date: Optional date string in YYYYMMDD format
        dropout_p: Optional dropout probability for MC Dropout (float)
        mc_samples: Optional number of MC samples for MC Dropout (int)
        n_nets: Optional number of nets for Deep Ensemble (int)
    
    Returns:
        tuple: (stats_df, fig) - DataFrame with statistics and matplotlib figure
    """
    stats_df = pd.DataFrame({
        'Tau': tau_values,
        'Distribution': [distribution] * len(tau_values),
        'Avg_Aleatoric_Entropy_norm': avg_ale_entropy_list,
        'Avg_Epistemic_Entropy_norm': avg_epi_entropy_list,
        'Avg_Total_Entropy_norm': avg_tot_entropy_list,
        'Correlation_Epi_Ale': correlation_list,
        'MSE': mse_list
    })
    
    base_filename = f"uncertainties_summary_{function_name}_{noise_type}_{distribution}_entropy"
    
    param_parts = []
    if model_name:
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                param_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                param_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                param_parts.append(f"K{n_nets}")
        
        if param_parts:
            model_prefix = f"{model_name}_{'_'.join(param_parts)}"
        else:
            model_prefix = model_name
        
        filename = f"{model_prefix}_{base_filename}"
    else:
        filename = base_filename
    
    if date:
        filename = f"{date}_{filename}"
    
    save_statistics(stats_df, filename, 
                    subfolder=f"{noise_type}/{func_type}", save_excel=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    title_suffix = f" - {model_name}" if model_name else ""
    
    # Plot 1: Average Entropies
    ax1.plot(tau_values, avg_ale_entropy_list, 'o-', linewidth=2, markersize=8, 
             label='Aleatoric Entropy', color='green')
    ax1.plot(tau_values, avg_epi_entropy_list, 's-', linewidth=2, markersize=8, 
             label='Epistemic Entropy', color='orange')
    ax1.plot(tau_values, avg_tot_entropy_list, '^-', linewidth=2, markersize=8, 
             label='Total Entropy', color='blue')
    ax1.set_xlabel('Tau (Noise Level)', fontsize=12)
    ax1.set_ylabel('Normalized Average Entropy', fontsize=12)
    ax1.set_title(f'Normalized Average Entropies vs Noise Level\n{function_name} Function ({noise_type.capitalize()}, {distribution}){title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation
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
    ax3.set_title(f'MSE vs Noise Level\n{function_name} Function ({noise_type.capitalize()}, {distribution}){title_suffix}', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    
    save_plot(fig, filename, 
              subfolder=f"{noise_type}/{func_type}")
    
    return stats_df, fig


def save_summary_statistics_entropy_ood(avg_ale_entropy_list, avg_epi_entropy_list, 
                                       avg_tot_entropy_list, correlation_list, mse_list,
                                       function_name, noise_type='heteroscedastic', 
                                       func_type='', model_name='', region_type='ID',
                                       date=None, dropout_p=None, mc_samples=None, n_nets=None,
                                       nll_list=None, crps_list=None,
                                       spearman_aleatoric_list=None, spearman_epistemic_list=None):
    """Helper function to save normalized entropy-based summary statistics for OOD experiments
    
    Args:
        avg_ale_entropy_list: List of normalized average aleatoric entropies (in [0, 1])
        avg_epi_entropy_list: List of normalized average epistemic entropies (in [0, 1])
        avg_tot_entropy_list: List of normalized average total entropies (in [0, 1])
        correlation_list: List of correlations between epistemic and aleatoric uncertainties
        mse_list: List of MSE values
        function_name: Name of the function (e.g., 'Linear', 'Sinusoidal')
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        model_name: Name of the model (e.g., 'MC_Dropout', 'Deep_Ensemble') - optional
        region_type: Region type ('ID', 'OOD', or 'Combined')
        date: Optional date string in YYYYMMDD format
        dropout_p: Optional dropout probability for MC Dropout (float)
        mc_samples: Optional number of MC samples for MC Dropout (int)
        n_nets: Optional number of nets for Deep Ensemble (int)
        nll_list: Optional list of NLL values
        crps_list: Optional list of CRPS values
        spearman_aleatoric_list: Optional list of Spearman correlations (aleatoric)
        spearman_epistemic_list: Optional list of Spearman correlations (epistemic)
        nll_list: Optional list of NLL values
        crps_list: Optional list of CRPS values
        spearman_aleatoric_list: Optional list of Spearman correlations (aleatoric)
        spearman_epistemic_list: Optional list of Spearman correlations (epistemic)
    
    Returns:
        tuple: (stats_df, None) - DataFrame with statistics (plots removed, only uncertainty plots with data points are displayed)
    """
    stats_dict = {
        'Avg_Aleatoric_Entropy_norm': avg_ale_entropy_list,
        'Avg_Epistemic_Entropy_norm': avg_epi_entropy_list,
        'Avg_Total_Entropy_norm': avg_tot_entropy_list,
        'Correlation_Epi_Ale': correlation_list,
        'MSE': mse_list
    }
    
    # Always add new metrics columns (use provided values or None)
    stats_dict['NLL'] = nll_list if nll_list is not None else [None] * len(avg_ale_entropy_list)
    stats_dict['CRPS'] = crps_list if crps_list is not None else [None] * len(avg_ale_entropy_list)
    stats_dict['Spearman_Aleatoric'] = spearman_aleatoric_list if spearman_aleatoric_list is not None else [None] * len(avg_ale_entropy_list)
    stats_dict['Spearman_Epistemic'] = spearman_epistemic_list if spearman_epistemic_list is not None else [None] * len(avg_ale_entropy_list)
    
    stats_df = pd.DataFrame(stats_dict)
    
    base_filename = f"{region_type}_uncertainties_summary_{function_name}_{noise_type}_entropy"
    
    param_parts = []
    if model_name:
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                param_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                param_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                param_parts.append(f"K{n_nets}")
        
        if param_parts:
            model_prefix = f"{model_name}_{'_'.join(param_parts)}"
        else:
            model_prefix = model_name
        
        filename = f"{model_prefix}_{base_filename}"
    else:
        filename = base_filename
    
    if date:
        filename = f"{date}_{filename}"
    
    save_statistics(stats_df, filename, 
                    subfolder=f"ood/{noise_type}/{func_type}", save_excel=True)
    
    # Summary plots removed - only uncertainty plots with data points are displayed
    # All statistics are printed to console and saved to Excel/CSV
    
    return stats_df, None


def save_summary_statistics_entropy_undersampling(avg_ale_entropy_list, avg_epi_entropy_list, 
                                                  avg_tot_entropy_list, correlation_list, mse_list,
                                                  function_name, noise_type='heteroscedastic', 
                                                  func_type='', model_name='', region_name='Region',
                                                  date=None, dropout_p=None, mc_samples=None, n_nets=None,
                                                  density_factor=None):
    """Helper function to save normalized entropy-based summary statistics for undersampling experiments
    
    Args:
        avg_ale_entropy_list: List of normalized average aleatoric entropies (in [0, 1])
        avg_epi_entropy_list: List of normalized average epistemic entropies (in [0, 1])
        avg_tot_entropy_list: List of normalized average total entropies (in [0, 1])
        correlation_list: List of correlations between epistemic and aleatoric uncertainties
        mse_list: List of MSE values
        function_name: Name of the function (e.g., 'Linear', 'Sinusoidal')
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        model_name: Name of the model (e.g., 'MC_Dropout', 'Deep_Ensemble') - optional
        region_name: Region name (e.g., 'Region_1', 'Undersampled', 'Well_sampled')
        date: Optional date string in YYYYMMDD format
        dropout_p: Optional dropout probability for MC Dropout (float)
        mc_samples: Optional number of MC samples for MC Dropout (int)
        n_nets: Optional number of nets for Deep Ensemble (int)
        density_factor: Optional density factor for filename
    
    Returns:
        tuple: (stats_df, fig) - DataFrame with statistics and matplotlib figure
    """
    stats_dict = {
        'Avg_Aleatoric_Entropy_norm': avg_ale_entropy_list,
        'Avg_Epistemic_Entropy_norm': avg_epi_entropy_list,
        'Avg_Total_Entropy_norm': avg_tot_entropy_list,
        'Correlation_Epi_Ale': correlation_list,
        'MSE': mse_list
    }
    
    stats_df = pd.DataFrame(stats_dict)
    
    base_filename = f"{region_name}_uncertainties_summary_{function_name}_{noise_type}_entropy"
    if density_factor is not None:
        base_filename = f"{region_name}_density{density_factor}_uncertainties_summary_{function_name}_{noise_type}_entropy"
    
    param_parts = []
    if model_name:
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                param_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                param_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                param_parts.append(f"K{n_nets}")
        
        if param_parts:
            model_prefix = f"{model_name}_{'_'.join(param_parts)}"
        else:
            model_prefix = model_name
        
        filename = f"{model_prefix}_{base_filename}"
    else:
        filename = base_filename
    
    if date:
        filename = f"{date}_{filename}"
    
    save_statistics(stats_df, filename, 
                    subfolder=f"undersampling/{noise_type}/{func_type}", save_excel=True)
    
    # Skip plot creation - only return statistics DataFrame
    return stats_df, None


# ========== Model Outputs Saving Function ==========

def save_model_outputs(mu_samples, sigma2_samples, x_grid, y_grid_clean,
                      x_train_subset=None, y_train_subset=None,
                      model_name='', noise_type='heteroscedastic', func_type='',
                      subfolder='', pct=None, tau=None, distribution=None,
                      dropout_p=None, mc_samples=None, n_nets=None,
                      date=None, **kwargs):
    """
    Save raw model outputs (mu_samples, sigma2_samples) to compressed numpy file.
    
    This allows later recomputation of uncertainties without rerunning models.
    
    Args:
        mu_samples: Raw mean predictions array [M/K/S, N] or [N, S] for BAMLSS
        sigma2_samples: Raw variance predictions array [M/K/S, N] or [N, S] for BAMLSS
        x_grid: Grid points used for prediction [N, 1] or [N]
        y_grid_clean: Clean function values at grid points [N, 1] or [N]
        x_train_subset: Optional training data subset [n_train, 1] or [n_train]
        y_train_subset: Optional training targets subset [n_train, 1] or [n_train]
        model_name: Model name (e.g., 'MC_Dropout', 'Deep_Ensemble', 'BNN', 'BAMLSS')
        noise_type: 'homoscedastic' or 'heteroscedastic'
        func_type: Function type identifier (e.g., 'linear', 'sin')
        subfolder: Additional subfolder path (e.g., 'sample_size', 'ood', 'noise_level')
        pct: Percentage for sample size experiments (optional)
        tau: Tau value for noise level experiments (optional)
        distribution: Distribution for noise level experiments (optional, e.g., 'normal', 'laplace')
        dropout_p: Dropout probability for MC Dropout (optional)
        mc_samples: Number of MC samples for MC Dropout (optional)
        n_nets: Number of nets for Deep Ensemble (optional)
        date: Optional date string in YYYYMMDD format
        **kwargs: Additional metadata to save
    
    Returns:
        Path: Filepath to saved .npz file
    """
    global outputs_dir
    
    # Determine outputs directory
    if outputs_dir is None:
        # Try to infer from plots_dir or stats_dir if set
        if plots_dir is not None:
            # outputs_dir should be parallel to plots_dir
            outputs_dir = plots_dir.parent / "outputs"
        elif stats_dir is not None:
            outputs_dir = stats_dir.parent / "outputs"
        else:
            # Use default directory structure
            current = Path.cwd()
            if current.name == 'Experiments':
                project_root = current.parent
            else:
                project_root = current
            
            # Try to determine experiment type from subfolder
            if 'sample_size' in subfolder:
                outputs_dir = project_root / "results" / "sample_size" / "outputs"
            elif 'ood' in subfolder:
                outputs_dir = project_root / "results" / "ood" / "outputs"
            elif 'noise_level' in subfolder:
                outputs_dir = project_root / "results" / "noise_level" / "outputs"
            else:
                # Default to sample_size
                outputs_dir = project_root / "results" / "sample_size" / "outputs"
    
    # Build subfolder path
    path_parts = []
    if noise_type:
        path_parts.append(noise_type)
    if func_type:
        path_parts.append(func_type)
    if distribution:
        path_parts.append(distribution)
    
    if path_parts:
        save_dir = outputs_dir / subfolder / "/".join(path_parts)
    else:
        save_dir = outputs_dir / subfolder if subfolder else outputs_dir
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    filename_parts = []
    if date:
        filename_parts.append(date)
    if model_name:
        filename_parts.append(model_name)
        if model_name == 'MC_Dropout':
            if dropout_p is not None:
                filename_parts.append(f"p{dropout_p}")
            if mc_samples is not None:
                filename_parts.append(f"M{mc_samples}")
        elif model_name == 'Deep_Ensemble':
            if n_nets is not None:
                filename_parts.append(f"K{n_nets}")
    
    if pct is not None:
        filename_parts.append(f"pct{pct}")
    if tau is not None:
        filename_parts.append(f"tau{tau}")
    if distribution:
        filename_parts.append(distribution)
    
    filename_parts.append("raw_outputs")
    filename = "_".join(filename_parts)
    filename = sanitize_filename(filename)
    
    filepath = save_dir / f"{filename}.npz"
    
    # Ensure arrays are numpy arrays and handle BAMLSS transpose if needed
    mu_samples = np.asarray(mu_samples)
    sigma2_samples = np.asarray(sigma2_samples)
    x_grid = np.asarray(x_grid)
    y_grid_clean = np.asarray(y_grid_clean)
    
    # BAMLSS returns [N, S] but we want [S, N] for consistency
    if model_name == 'BAMLSS' and mu_samples.shape[0] != mu_samples.shape[1]:
        # Check if it's [N, S] format and transpose
        if mu_samples.shape[1] < mu_samples.shape[0] and len(x_grid) == mu_samples.shape[0]:
            mu_samples = mu_samples.T
            sigma2_samples = sigma2_samples.T
    
    # Prepare data dictionary
    save_dict = {
        'mu_samples': mu_samples,
        'sigma2_samples': sigma2_samples,
        'x_grid': x_grid,
        'y_grid_clean': y_grid_clean,
    }
    
    if x_train_subset is not None:
        save_dict['x_train_subset'] = np.asarray(x_train_subset)
    if y_train_subset is not None:
        save_dict['y_train_subset'] = np.asarray(y_train_subset)
    
    # Add metadata as arrays (for npz compatibility)
    save_dict['model_name'] = np.array([model_name], dtype=object) if model_name else None
    save_dict['noise_type'] = np.array([noise_type], dtype=object) if noise_type else None
    save_dict['func_type'] = np.array([func_type], dtype=object) if func_type else None
    
    if pct is not None:
        save_dict['pct'] = np.array([pct], dtype=np.float32)
    if tau is not None:
        save_dict['tau'] = np.array([tau], dtype=np.float32)
    if distribution:
        save_dict['distribution'] = np.array([distribution], dtype=object)
    if dropout_p is not None:
        save_dict['dropout_p'] = np.array([dropout_p], dtype=np.float32)
    if mc_samples is not None:
        save_dict['mc_samples'] = np.array([mc_samples], dtype=np.int32)
    if n_nets is not None:
        save_dict['n_nets'] = np.array([n_nets], dtype=np.int32)
    if date:
        save_dict['date'] = np.array([date], dtype=object)
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, (str,)):
                save_dict[key] = np.array([value], dtype=object)
            elif isinstance(value, (int,)):
                save_dict[key] = np.array([value], dtype=np.int32)
            elif isinstance(value, (float,)):
                save_dict[key] = np.array([value], dtype=np.float32)
            else:
                save_dict[key] = np.asarray(value)
    
    # Remove None values
    save_dict = {k: v for k, v in save_dict.items() if v is not None}
    
    # Save compressed
    np.savez_compressed(filepath, **save_dict)
    print(f"Saved model outputs: {filepath}")
    
    return filepath