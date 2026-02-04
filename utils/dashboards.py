"""
Dashboard generation module for experiment results visualization.

This module provides functions to create comprehensive dashboards that combine
multiple plots and statistics tables for better analysis and presentation of
experiment results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from utils.results_save import sanitize_filename, _get_default_dirs

# Module-level variable for dashboard directory
dashboards_dir = None


def _get_dashboards_dir():
    """Get or create the dashboards directory"""
    global dashboards_dir
    if dashboards_dir is None:
        # Try to infer from results structure
        current = Path.cwd()
        if current.name == 'Experiments':
            project_root = current.parent
        else:
            project_root = current
        
        dashboards_dir = project_root / "results" / "dashboards"
        dashboards_dir.mkdir(parents=True, exist_ok=True)
    
    return dashboards_dir


def save_dashboard(fig, filename, experiment_type, noise_type='heteroscedastic', 
                   date=None, distribution=None, subfolder=''):
    """Save a dashboard figure to the dashboards directory"""
    dash_dir = _get_dashboards_dir()
    
    # Build subfolder path
    path_parts = [experiment_type, noise_type]
    if distribution:
        path_parts.append(distribution)
    if subfolder:
        path_parts.append(subfolder)
    
    save_dir = dash_dir / "/".join(path_parts)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    if date:
        filename = f"{date}_{filename}"
    
    filepath = save_dir / f"{sanitize_filename(filename)}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved dashboard: {filepath}")
    return filepath


def create_statistics_table(stats_df, title, figsize=(14, 8), highlight_extremes=True):
    """
    Create a formatted statistics table visualization.
    
    Args:
        stats_df: DataFrame with statistics
        title: Title for the table
        figsize: Figure size tuple
        highlight_extremes: Whether to highlight extreme values
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    headers = list(stats_df.columns)
    
    for idx, row in stats_df.iterrows():
        table_data.append([str(val) if pd.notna(val) else 'N/A' for val in row.values])
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     rowLabels=[str(idx) for idx in stats_df.index],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight extreme values if requested
    if highlight_extremes and len(stats_df) > 0:
        numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
        for col_idx, col in enumerate(stats_df.columns):
            if col in numeric_cols:
                col_values = pd.to_numeric(stats_df[col], errors='coerce')
                if len(col_values.dropna()) > 0:
                    min_val = col_values.min()
                    max_val = col_values.max()
                    for row_idx in range(len(stats_df)):
                        val = col_values.iloc[row_idx]
                        if pd.notna(val):
                            if val == min_val:
                                table[(row_idx + 1, col_idx)].set_facecolor('#E8F5E9')
                            elif val == max_val:
                                table[(row_idx + 1, col_idx)].set_facecolor('#FFEBEE')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def create_model_comparison_plot_sample_size(accumulated_stats, noise_type, func_type, date=None):
    """
    Create a multi-panel comparison plot for sample size experiments.
    
    Args:
        accumulated_stats: Dictionary with structure {(model_name, func_type): {'variance': df, 'entropy': df}}
        noise_type: Type of noise
        func_type: Function type ('linear' or 'sin')
        date: Optional date string
    
    Returns:
        matplotlib Figure object
    """
    # Filter stats for this function type
    relevant_stats = {k: v for k, v in accumulated_stats.items() if k[1] == func_type}
    
    if not relevant_stats:
        return None
    
    # Determine number of subplots needed
    has_mse = False
    has_extra_metrics = False
    
    for stats_data in relevant_stats.values():
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0:
            if 'MSE' in variance_df.columns:
                has_mse = True
            if any(col in variance_df.columns for col in ['NLL', 'CRPS', 'Spearman_Aleatoric', 'Spearman_Epistemic']):
                has_extra_metrics = True
            break
    
    # Create subplots: 3 rows x 2 cols = 6 subplots
    # Row 1: Aleatoric, Epistemic
    # Row 2: Total, Correlation
    # Row 3: MSE, Extra metrics (if available)
    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 18))
    axes = axes.flatten()
    
    # Get x-axis values (percentages) from first available dataframe
    x_values = None
    for stats_data in relevant_stats.values():
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Percentage' in variance_df.columns:
            x_values = variance_df['Percentage'].values
            break
    
    if x_values is None:
        return None
    
    # Define colors for different models
    model_colors = {
        'MC_Dropout': '#1f77b4',
        'Deep_Ensemble': '#ff7f0e',
        'BNN': '#2ca02c',
        'BAMLSS': '#d62728'
    }
    
    # Plot 1: Aleatoric Uncertainty
    ax = axes[0]
    for (model_name, _), stats_data in sorted(relevant_stats.items()):
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Avg_Aleatoric_norm' in variance_df.columns:
            color = model_colors.get(model_name, '#9467bd')
            ax.plot(x_values, variance_df['Avg_Aleatoric_norm'].values, 
                   'o-', linewidth=2, markersize=6, label=model_name, color=color)
    ax.set_xlabel('Training Data Percentage (%)', fontsize=11)
    ax.set_ylabel('Normalized Aleatoric Uncertainty', fontsize=11)
    ax.set_title('Aleatoric Uncertainty vs Training Data Percentage', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    
    # Plot 2: Epistemic Uncertainty
    ax = axes[1]
    for (model_name, _), stats_data in sorted(relevant_stats.items()):
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Avg_Epistemic_norm' in variance_df.columns:
            color = model_colors.get(model_name, '#9467bd')
            ax.plot(x_values, variance_df['Avg_Epistemic_norm'].values, 
                   's-', linewidth=2, markersize=6, label=model_name, color=color)
    ax.set_xlabel('Training Data Percentage (%)', fontsize=11)
    ax.set_ylabel('Normalized Epistemic Uncertainty', fontsize=11)
    ax.set_title('Epistemic Uncertainty vs Training Data Percentage', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    
    # Plot 3: Total Uncertainty
    ax = axes[2]
    for (model_name, _), stats_data in sorted(relevant_stats.items()):
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Avg_Total_norm' in variance_df.columns:
            color = model_colors.get(model_name, '#9467bd')
            ax.plot(x_values, variance_df['Avg_Total_norm'].values, 
                   '^-', linewidth=2, markersize=6, label=model_name, color=color)
    ax.set_xlabel('Training Data Percentage (%)', fontsize=11)
    ax.set_title('Total Uncertainty vs Training Data Percentage', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    
    # Plot 4: Correlation
    ax = axes[3]
    for (model_name, _), stats_data in sorted(relevant_stats.items()):
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Correlation_Epi_Ale' in variance_df.columns:
            color = model_colors.get(model_name, '#9467bd')
            ax.plot(x_values, variance_df['Correlation_Epi_Ale'].values, 
                   'D-', linewidth=2, markersize=6, label=model_name, color=color)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Training Data Percentage (%)', fontsize=11)
    ax.set_ylabel('Correlation Coefficient', fontsize=11)
    ax.set_title('Correlation: Epistemic vs Aleatoric', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    
    # Plot 5: MSE (if available)
    ax = axes[4]
    if has_mse:
        for (model_name, _), stats_data in sorted(relevant_stats.items()):
            variance_df = stats_data.get('variance')
            if variance_df is not None and len(variance_df) > 0 and 'MSE' in variance_df.columns:
                mse_values = variance_df['MSE'].values
                if not np.all(pd.isna(mse_values)):
                    color = model_colors.get(model_name, '#9467bd')
                    ax.plot(x_values, mse_values, 
                           '*-', linewidth=2, markersize=6, label=model_name, color=color)
        ax.set_xlabel('Training Data Percentage (%)', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title('Mean Squared Error vs Training Data Percentage', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_yscale('log')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'MSE data not available', 
               ha='center', va='center', fontsize=12, style='italic')
    
    # Plot 6: Extra metrics (NLL, CRPS, Spearman) if available
    ax = axes[5]
    if has_extra_metrics:
        metric_plots = []
        for (model_name, _), stats_data in sorted(relevant_stats.items()):
            variance_df = stats_data.get('variance')
            if variance_df is not None and len(variance_df) > 0:
                color = model_colors.get(model_name, '#9467bd')
                # Plot NLL if available
                if 'NLL' in variance_df.columns:
                    nll_values = variance_df['NLL'].values
                    if not np.all(pd.isna(nll_values)):
                        ax.plot(x_values, nll_values, 'o-', linewidth=2, markersize=4, 
                               label=f'{model_name} (NLL)', color=color, alpha=0.7)
                # Plot CRPS if available
                if 'CRPS' in variance_df.columns:
                    crps_values = variance_df['CRPS'].values
                    if not np.all(pd.isna(crps_values)):
                        ax.plot(x_values, crps_values, 's--', linewidth=2, markersize=4, 
                               label=f'{model_name} (CRPS)', color=color, alpha=0.7)
        if ax.get_legend_handles_labels()[0]:
            ax.set_xlabel('Training Data Percentage (%)', fontsize=11)
            ax.set_ylabel('Metric Value', fontsize=11)
            ax.set_title('Additional Metrics (NLL, CRPS)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 105)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Additional metrics not available', 
                   ha='center', va='center', fontsize=12, style='italic')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Additional metrics not available', 
               ha='center', va='center', fontsize=12, style='italic')
    
    # Add overall title
    func_name_map = {'linear': 'Linear', 'sin': 'Sinusoidal'}
    func_name = func_name_map.get(func_type, func_type.capitalize())
    title = f'Sample Size Experiment Dashboard - {func_name} Function ({noise_type.capitalize()} Noise)'
    if date:
        title += f' - {date}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def create_model_comparison_plot_noise_level(accumulated_stats, noise_type, distribution, func_type, date=None):
    """
    Create a multi-panel comparison plot for noise level experiments.
    
    Args:
        accumulated_stats: Dictionary with structure {(model_name, func_type): {'variance': df, 'entropy': df}}
        noise_type: Type of noise
        distribution: Distribution type ('normal' or 'laplace')
        func_type: Function type ('linear' or 'sin')
        date: Optional date string
    
    Returns:
        matplotlib Figure object
    """
    # Filter stats for this function type
    relevant_stats = {k: v for k, v in accumulated_stats.items() if k[1] == func_type}
    
    if not relevant_stats:
        return None
    
    # Get x-axis values (tau) from first available dataframe
    x_values = None
    for stats_data in relevant_stats.values():
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Tau' in variance_df.columns:
            x_values = variance_df['Tau'].values
            break
    
    if x_values is None:
        return None
    
    # Determine if MSE and extra metrics are available
    has_mse = False
    has_extra_metrics = False
    
    for stats_data in relevant_stats.values():
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0:
            if 'MSE' in variance_df.columns:
                has_mse = True
            if any(col in variance_df.columns for col in ['NLL', 'CRPS', 'Spearman_Aleatoric', 'Spearman_Epistemic']):
                has_extra_metrics = True
            break
    
    # Create subplots
    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 18))
    axes = axes.flatten()
    
    # Define colors for different models
    model_colors = {
        'MC_Dropout': '#1f77b4',
        'Deep_Ensemble': '#ff7f0e',
        'BNN': '#2ca02c',
        'BAMLSS': '#d62728'
    }
    
    # Plot 1: Aleatoric Uncertainty
    ax = axes[0]
    for (model_name, _), stats_data in sorted(relevant_stats.items()):
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Avg_Aleatoric_norm' in variance_df.columns:
            color = model_colors.get(model_name, '#9467bd')
            ax.plot(x_values, variance_df['Avg_Aleatoric_norm'].values, 
                   'o-', linewidth=2, markersize=6, label=model_name, color=color)
    ax.set_xlabel('Tau (Noise Level)', fontsize=11)
    ax.set_ylabel('Normalized Aleatoric Uncertainty', fontsize=11)
    ax.set_title('Aleatoric Uncertainty vs Tau', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Epistemic Uncertainty
    ax = axes[1]
    for (model_name, _), stats_data in sorted(relevant_stats.items()):
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Avg_Epistemic_norm' in variance_df.columns:
            color = model_colors.get(model_name, '#9467bd')
            ax.plot(x_values, variance_df['Avg_Epistemic_norm'].values, 
                   's-', linewidth=2, markersize=6, label=model_name, color=color)
    ax.set_xlabel('Tau (Noise Level)', fontsize=11)
    ax.set_ylabel('Normalized Epistemic Uncertainty', fontsize=11)
    ax.set_title('Epistemic Uncertainty vs Tau', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total Uncertainty
    ax = axes[2]
    for (model_name, _), stats_data in sorted(relevant_stats.items()):
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Avg_Total_norm' in variance_df.columns:
            color = model_colors.get(model_name, '#9467bd')
            ax.plot(x_values, variance_df['Avg_Total_norm'].values, 
                   '^-', linewidth=2, markersize=6, label=model_name, color=color)
    ax.set_xlabel('Tau (Noise Level)', fontsize=11)
    ax.set_ylabel('Normalized Total Uncertainty', fontsize=11)
    ax.set_title('Total Uncertainty vs Tau', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Correlation
    ax = axes[3]
    for (model_name, _), stats_data in sorted(relevant_stats.items()):
        variance_df = stats_data.get('variance')
        if variance_df is not None and len(variance_df) > 0 and 'Correlation_Epi_Ale' in variance_df.columns:
            color = model_colors.get(model_name, '#9467bd')
            ax.plot(x_values, variance_df['Correlation_Epi_Ale'].values, 
                   'D-', linewidth=2, markersize=6, label=model_name, color=color)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Tau (Noise Level)', fontsize=11)
    ax.set_ylabel('Correlation Coefficient', fontsize=11)
    ax.set_title('Correlation: Epistemic vs Aleatoric', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: MSE (if available)
    ax = axes[4]
    if has_mse:
        for (model_name, _), stats_data in sorted(relevant_stats.items()):
            variance_df = stats_data.get('variance')
            if variance_df is not None and len(variance_df) > 0 and 'MSE' in variance_df.columns:
                mse_values = variance_df['MSE'].values
                if not np.all(pd.isna(mse_values)):
                    color = model_colors.get(model_name, '#9467bd')
                    ax.plot(x_values, mse_values, 
                           '*-', linewidth=2, markersize=6, label=model_name, color=color)
        ax.set_xlabel('Tau (Noise Level)', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title('Mean Squared Error vs Tau', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'MSE data not available', 
               ha='center', va='center', fontsize=12, style='italic')
    
    # Plot 6: Extra metrics
    ax = axes[5]
    if has_extra_metrics:
        for (model_name, _), stats_data in sorted(relevant_stats.items()):
            variance_df = stats_data.get('variance')
            if variance_df is not None and len(variance_df) > 0:
                color = model_colors.get(model_name, '#9467bd')
                if 'NLL' in variance_df.columns:
                    nll_values = variance_df['NLL'].values
                    if not np.all(pd.isna(nll_values)):
                        ax.plot(x_values, nll_values, 'o-', linewidth=2, markersize=4, 
                               label=f'{model_name} (NLL)', color=color, alpha=0.7)
                if 'CRPS' in variance_df.columns:
                    crps_values = variance_df['CRPS'].values
                    if not np.all(pd.isna(crps_values)):
                        ax.plot(x_values, crps_values, 's--', linewidth=2, markersize=4, 
                               label=f'{model_name} (CRPS)', color=color, alpha=0.7)
        if ax.get_legend_handles_labels()[0]:
            ax.set_xlabel('Tau (Noise Level)', fontsize=11)
            ax.set_ylabel('Metric Value', fontsize=11)
            ax.set_title('Additional Metrics (NLL, CRPS)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Additional metrics not available', 
                   ha='center', va='center', fontsize=12, style='italic')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Additional metrics not available', 
               ha='center', va='center', fontsize=12, style='italic')
    
    # Add overall title
    func_name_map = {'linear': 'Linear', 'sin': 'Sinusoidal'}
    func_name = func_name_map.get(func_type, func_type.capitalize())
    title = f'Noise Level Experiment Dashboard - {func_name} Function ({noise_type.capitalize()} Noise, {distribution.capitalize()} Distribution)'
    if date:
        title += f' - {date}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def create_experiment_dashboard(experiment_type, accumulated_stats, noise_type='heteroscedastic', 
                                date=None, distribution=None, **kwargs):
    """
    Main function to create comprehensive dashboards for experiment results.
    
    Args:
        experiment_type: Type of experiment ('sample_size', 'noise_level', 'ood', 'undersampling')
        accumulated_stats: Dictionary with accumulated statistics
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        date: Optional date string in YYYYMMDD format
        distribution: Distribution type for noise_level experiments ('normal' or 'laplace')
        **kwargs: Additional arguments
    
    Returns:
        List of saved file paths
    """
    if not accumulated_stats:
        print(f"No statistics available for {experiment_type} dashboard")
        return []
    
    saved_paths = []
    
    if experiment_type == 'sample_size':
        # Create dashboards for each function type
        func_types = set()
        for (model_name, func_type) in accumulated_stats.keys():
            func_types.add(func_type)
        
        for func_type in sorted(func_types):
            fig = create_model_comparison_plot_sample_size(
                accumulated_stats, noise_type, func_type, date
            )
            if fig is not None:
                filename = f"sample_size_{noise_type}_{func_type}_dashboard"
                filepath = save_dashboard(fig, filename, experiment_type, noise_type, date)
                saved_paths.append(filepath)
                plt.close(fig)
            
            # Create statistics tables for each model
            for (model_name, ft), stats_data in accumulated_stats.items():
                if ft != func_type:
                    continue
                variance_df = stats_data.get('variance')
                if variance_df is not None and len(variance_df) > 0:
                    func_name_map = {'linear': 'Linear', 'sin': 'Sinusoidal'}
                    func_name = func_name_map.get(ft, ft.capitalize())
                    table_title = f'{model_name} - {func_name} Function Statistics ({noise_type.capitalize()} Noise)'
                    table_fig = create_statistics_table(variance_df, table_title)
                    
                    # Save table
                    dash_dir = _get_dashboards_dir()
                    table_dir = dash_dir / experiment_type / noise_type / "statistics_tables"
                    table_dir.mkdir(parents=True, exist_ok=True)
                    
                    table_filename = f"{date}_{model_name}_{ft}_statistics_table" if date else f"{model_name}_{ft}_statistics_table"
                    table_filepath = table_dir / f"{sanitize_filename(table_filename)}.png"
                    table_fig.savefig(table_filepath, dpi=300, bbox_inches='tight')
                    print(f"Saved statistics table: {table_filepath}")
                    saved_paths.append(table_filepath)
                    plt.close(table_fig)
    
    elif experiment_type == 'noise_level':
        if distribution is None:
            print("Warning: distribution not specified for noise_level dashboard")
            return []
        
        # Create dashboards for each function type
        func_types = set()
        for (model_name, func_type) in accumulated_stats.keys():
            func_types.add(func_type)
        
        for func_type in sorted(func_types):
            fig = create_model_comparison_plot_noise_level(
                accumulated_stats, noise_type, distribution, func_type, date
            )
            if fig is not None:
                filename = f"noise_level_{noise_type}_{distribution}_{func_type}_dashboard"
                filepath = save_dashboard(fig, filename, experiment_type, noise_type, date, distribution)
                saved_paths.append(filepath)
                plt.close(fig)
            
            # Create statistics tables for each model
            for (model_name, ft), stats_data in accumulated_stats.items():
                if ft != func_type:
                    continue
                variance_df = stats_data.get('variance')
                if variance_df is not None and len(variance_df) > 0:
                    func_name_map = {'linear': 'Linear', 'sin': 'Sinusoidal'}
                    func_name = func_name_map.get(ft, ft.capitalize())
                    table_title = f'{model_name} - {func_name} Function Statistics ({noise_type.capitalize()} Noise, {distribution.capitalize()} Distribution)'
                    table_fig = create_statistics_table(variance_df, table_title)
                    
                    # Save table
                    dash_dir = _get_dashboards_dir()
                    table_dir = dash_dir / experiment_type / noise_type / distribution / "statistics_tables"
                    table_dir.mkdir(parents=True, exist_ok=True)
                    
                    table_filename = f"{date}_{model_name}_{ft}_statistics_table" if date else f"{model_name}_{ft}_statistics_table"
                    table_filepath = table_dir / f"{sanitize_filename(table_filename)}.png"
                    table_fig.savefig(table_filepath, dpi=300, bbox_inches='tight')
                    print(f"Saved statistics table: {table_filepath}")
                    saved_paths.append(table_filepath)
                    plt.close(table_fig)
    
    elif experiment_type == 'ood':
        # OOD dashboards - compare ID vs OOD vs Combined regions
        # Structure: {(model_name, func_type, region_type): {'variance': df, 'entropy': df}}
        
        # Group by function type and model
        func_types = set()
        models = set()
        region_types = ['ID', 'OOD', 'Combined']
        
        for (model_name, func_type, region_type) in accumulated_stats.keys():
            func_types.add(func_type)
            models.add(model_name)
        
        for func_type in sorted(func_types):
            # Create comparison plot for ID vs OOD vs Combined
            n_rows = 3
            n_cols = 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 18))
            axes = axes.flatten()
            
            # Define colors for different models
            model_colors = {
                'MC_Dropout': '#1f77b4',
                'Deep_Ensemble': '#ff7f0e',
                'BNN': '#2ca02c',
                'BAMLSS': '#d62728'
            }
            
            # Plot 1: Aleatoric Uncertainty by Region
            ax = axes[0]
            x_positions = np.arange(len(region_types))
            width = 0.2
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_type in region_types:
                    key = (model_name, func_type, region_type)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'Avg_Aleatoric_Variance' in variance_df.columns:
                            val = variance_df['Avg_Aleatoric_Variance'].iloc[0]
                            values.append(val if pd.notna(val) else 0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                offset = (idx - len(models)/2 + 0.5) * width
                ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            ax.set_ylabel('Aleatoric Uncertainty', fontsize=11)
            ax.set_title('Aleatoric Uncertainty by Region', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(region_types)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Epistemic Uncertainty by Region
            ax = axes[1]
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_type in region_types:
                    key = (model_name, func_type, region_type)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'Avg_Epistemic_Variance' in variance_df.columns:
                            val = variance_df['Avg_Epistemic_Variance'].iloc[0]
                            values.append(val if pd.notna(val) else 0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                offset = (idx - len(models)/2 + 0.5) * width
                ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            ax.set_ylabel('Epistemic Uncertainty', fontsize=11)
            ax.set_title('Epistemic Uncertainty by Region', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(region_types)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Total Uncertainty by Region
            ax = axes[2]
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_type in region_types:
                    key = (model_name, func_type, region_type)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'Avg_Total_Variance' in variance_df.columns:
                            val = variance_df['Avg_Total_Variance'].iloc[0]
                            values.append(val if pd.notna(val) else 0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                offset = (idx - len(models)/2 + 0.5) * width
                ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            ax.set_ylabel('Total Uncertainty', fontsize=11)
            ax.set_title('Total Uncertainty by Region', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(region_types)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Correlation by Region
            ax = axes[3]
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_type in region_types:
                    key = (model_name, func_type, region_type)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'Correlation_Epi_Ale' in variance_df.columns:
                            val = variance_df['Correlation_Epi_Ale'].iloc[0]
                            values.append(val if pd.notna(val) else 0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                offset = (idx - len(models)/2 + 0.5) * width
                ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylabel('Correlation', fontsize=11)
            ax.set_title('Correlation: Epistemic vs Aleatoric by Region', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(region_types)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 5: MSE by Region
            ax = axes[4]
            has_mse = False
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_type in region_types:
                    key = (model_name, func_type, region_type)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'MSE' in variance_df.columns:
                            val = variance_df['MSE'].iloc[0]
                            if pd.notna(val):
                                has_mse = True
                                values.append(val)
                            else:
                                values.append(0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                if any(v > 0 for v in values):
                    offset = (idx - len(models)/2 + 0.5) * width
                    ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            if has_mse:
                ax.set_ylabel('MSE', fontsize=11)
                ax.set_title('MSE by Region', fontsize=12, fontweight='bold')
                ax.set_xticks(x_positions)
                ax.set_xticklabels(region_types)
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_yscale('log')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'MSE data not available', 
                       ha='center', va='center', fontsize=12, style='italic')
            
            # Plot 6: Additional metrics if available
            ax = axes[5]
            has_extra = False
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_type in region_types:
                    key = (model_name, func_type, region_type)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'NLL' in variance_df.columns:
                            val = variance_df['NLL'].iloc[0]
                            if pd.notna(val):
                                has_extra = True
                                values.append(val)
                            else:
                                values.append(0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                if any(v > 0 for v in values):
                    offset = (idx - len(models)/2 + 0.5) * width
                    ax.bar(x_positions + offset, values, width, label=f'{model_name} (NLL)', color=color, alpha=0.7)
            if has_extra:
                ax.set_ylabel('NLL', fontsize=11)
                ax.set_title('NLL by Region', fontsize=12, fontweight='bold')
                ax.set_xticks(x_positions)
                ax.set_xticklabels(region_types)
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'Additional metrics not available', 
                       ha='center', va='center', fontsize=12, style='italic')
            
            # Add overall title
            func_name_map = {'linear': 'Linear', 'sin': 'Sinusoidal'}
            func_name = func_name_map.get(func_type, func_type.capitalize())
            title = f'OOD Experiment Dashboard - {func_name} Function ({noise_type.capitalize()} Noise)'
            if date:
                title += f' - {date}'
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            # Save dashboard
            filename = f"ood_{noise_type}_{func_type}_dashboard"
            filepath = save_dashboard(fig, filename, experiment_type, noise_type, date)
            saved_paths.append(filepath)
            plt.close(fig)
            
            # Create statistics tables for each model/region combination
            for (model_name, ft, region_type), stats_data in accumulated_stats.items():
                if ft != func_type:
                    continue
                variance_df = stats_data.get('variance')
                if variance_df is not None and len(variance_df) > 0:
                    func_name_map = {'linear': 'Linear', 'sin': 'Sinusoidal'}
                    func_name = func_name_map.get(ft, ft.capitalize())
                    table_title = f'{model_name} - {func_name} Function - {region_type} Region ({noise_type.capitalize()} Noise)'
                    table_fig = create_statistics_table(variance_df, table_title)
                    
                    # Save table
                    dash_dir = _get_dashboards_dir()
                    table_dir = dash_dir / experiment_type / noise_type / "statistics_tables"
                    table_dir.mkdir(parents=True, exist_ok=True)
                    
                    table_filename = f"{date}_{model_name}_{ft}_{region_type}_statistics_table" if date else f"{model_name}_{ft}_{region_type}_statistics_table"
                    table_filepath = table_dir / f"{sanitize_filename(table_filename)}.png"
                    table_fig.savefig(table_filepath, dpi=300, bbox_inches='tight')
                    print(f"Saved statistics table: {table_filepath}")
                    saved_paths.append(table_filepath)
                    plt.close(table_fig)
    
    elif experiment_type == 'undersampling':
        # Undersampling dashboards - region-based with density factors
        # Structure: {(model_name, func_type, region_name): {'variance': df, 'entropy': df}}
        
        # Group by function type
        func_types = set()
        models = set()
        regions = []
        
        for (model_name, func_type, region_name) in accumulated_stats.keys():
            func_types.add(func_type)
            models.add(model_name)
            if region_name not in regions:
                regions.append(region_name)
        
        regions = sorted(regions, key=lambda x: (x.startswith('Region'), int(x.split()[-1]) if x.split()[-1].isdigit() else 999))
        
        for func_type in sorted(func_types):
            # Create comparison plot
            n_rows = 3
            n_cols = 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 18))
            axes = axes.flatten()
            
            # Define colors for different models
            model_colors = {
                'MC_Dropout': '#1f77b4',
                'Deep_Ensemble': '#ff7f0e',
                'BNN': '#2ca02c',
                'BAMLSS': '#d62728'
            }
            
            x_positions = np.arange(len(regions))
            
            # Plot 1: Aleatoric Uncertainty by Region
            ax = axes[0]
            width = 0.2
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_name in regions:
                    key = (model_name, func_type, region_name)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'Avg_Aleatoric_norm' in variance_df.columns:
                            val = variance_df['Avg_Aleatoric_norm'].iloc[0]
                            values.append(val if pd.notna(val) else 0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                offset = (idx - len(models)/2 + 0.5) * width
                ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            ax.set_ylabel('Aleatoric Uncertainty', fontsize=11)
            ax.set_title('Aleatoric Uncertainty by Region', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Epistemic Uncertainty by Region
            ax = axes[1]
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_name in regions:
                    key = (model_name, func_type, region_name)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'Avg_Epistemic_norm' in variance_df.columns:
                            val = variance_df['Avg_Epistemic_norm'].iloc[0]
                            values.append(val if pd.notna(val) else 0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                offset = (idx - len(models)/2 + 0.5) * width
                ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            ax.set_ylabel('Epistemic Uncertainty', fontsize=11)
            ax.set_title('Epistemic Uncertainty by Region', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Total Uncertainty by Region
            ax = axes[2]
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_name in regions:
                    key = (model_name, func_type, region_name)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'Avg_Total_norm' in variance_df.columns:
                            val = variance_df['Avg_Total_norm'].iloc[0]
                            values.append(val if pd.notna(val) else 0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                offset = (idx - len(models)/2 + 0.5) * width
                ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            ax.set_ylabel('Total Uncertainty', fontsize=11)
            ax.set_title('Total Uncertainty by Region', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Correlation by Region
            ax = axes[3]
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_name in regions:
                    key = (model_name, func_type, region_name)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'Correlation_Epi_Ale' in variance_df.columns:
                            val = variance_df['Correlation_Epi_Ale'].iloc[0]
                            values.append(val if pd.notna(val) else 0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                offset = (idx - len(models)/2 + 0.5) * width
                ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylabel('Correlation', fontsize=11)
            ax.set_title('Correlation: Epistemic vs Aleatoric by Region', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 5: MSE by Region
            ax = axes[4]
            has_mse = False
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_name in regions:
                    key = (model_name, func_type, region_name)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'MSE' in variance_df.columns:
                            val = variance_df['MSE'].iloc[0]
                            if pd.notna(val):
                                has_mse = True
                                values.append(val)
                            else:
                                values.append(0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                if any(v > 0 for v in values):
                    offset = (idx - len(models)/2 + 0.5) * width
                    ax.bar(x_positions + offset, values, width, label=model_name, color=color, alpha=0.7)
            if has_mse:
                ax.set_ylabel('MSE', fontsize=11)
                ax.set_title('MSE by Region', fontsize=12, fontweight='bold')
                ax.set_xticks(x_positions)
                ax.set_xticklabels(regions, rotation=45, ha='right')
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_yscale('log')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'MSE data not available', 
                       ha='center', va='center', fontsize=12, style='italic')
            
            # Plot 6: Additional metrics if available
            ax = axes[5]
            has_extra = False
            for idx, model_name in enumerate(sorted(models)):
                color = model_colors.get(model_name, '#9467bd')
                values = []
                for region_name in regions:
                    key = (model_name, func_type, region_name)
                    if key in accumulated_stats:
                        stats_data = accumulated_stats[key]
                        variance_df = stats_data.get('variance')
                        if variance_df is not None and len(variance_df) > 0 and 'NLL' in variance_df.columns:
                            val = variance_df['NLL'].iloc[0]
                            if pd.notna(val):
                                has_extra = True
                                values.append(val)
                            else:
                                values.append(0)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                if any(v > 0 for v in values):
                    offset = (idx - len(models)/2 + 0.5) * width
                    ax.bar(x_positions + offset, values, width, label=f'{model_name} (NLL)', color=color, alpha=0.7)
            if has_extra:
                ax.set_ylabel('NLL', fontsize=11)
                ax.set_title('NLL by Region', fontsize=12, fontweight='bold')
                ax.set_xticks(x_positions)
                ax.set_xticklabels(regions, rotation=45, ha='right')
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'Additional metrics not available', 
                       ha='center', va='center', fontsize=12, style='italic')
            
            # Add overall title
            func_name_map = {'linear': 'Linear', 'sin': 'Sinusoidal'}
            func_name = func_name_map.get(func_type, func_type.capitalize())
            title = f'Undersampling Experiment Dashboard - {func_name} Function ({noise_type.capitalize()} Noise)'
            if date:
                title += f' - {date}'
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            # Save dashboard
            filename = f"undersampling_{noise_type}_{func_type}_dashboard"
            filepath = save_dashboard(fig, filename, experiment_type, noise_type, date)
            saved_paths.append(filepath)
            plt.close(fig)
            
            # Create statistics tables for each model/region combination
            for (model_name, ft, region_name), stats_data in accumulated_stats.items():
                if ft != func_type:
                    continue
                variance_df = stats_data.get('variance')
                if variance_df is not None and len(variance_df) > 0:
                    func_name_map = {'linear': 'Linear', 'sin': 'Sinusoidal'}
                    func_name = func_name_map.get(ft, ft.capitalize())
                    table_title = f'{model_name} - {func_name} Function - {region_name} ({noise_type.capitalize()} Noise)'
                    table_fig = create_statistics_table(variance_df, table_title)
                    
                    # Save table
                    dash_dir = _get_dashboards_dir()
                    table_dir = dash_dir / experiment_type / noise_type / "statistics_tables"
                    table_dir.mkdir(parents=True, exist_ok=True)
                    
                    table_filename = f"{date}_{model_name}_{ft}_{region_name}_statistics_table" if date else f"{model_name}_{ft}_{region_name}_statistics_table"
                    table_filepath = table_dir / f"{sanitize_filename(table_filename)}.png"
                    table_fig.savefig(table_filepath, dpi=300, bbox_inches='tight')
                    print(f"Saved statistics table: {table_filepath}")
                    saved_paths.append(table_filepath)
                    plt.close(table_fig)
    
    else:
        print(f"Unknown experiment type: {experiment_type}")
    
    return saved_paths

