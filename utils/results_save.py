## Utility functions for saving results
import pandas as pd
import matplotlib.pyplot as plt


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
    if subfolder:
        save_dir = plots_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = plots_dir
    
    filepath = save_dir / f"{sanitize_filename(filename)}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    return filepath

def save_statistics(data_dict, filename, subfolder='', save_excel=True):
    """Save statistics dictionary to CSV and optionally Excel"""
    if subfolder:
        save_dir = stats_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = stats_dir
    
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
    if subfolder:
        save_dir = stats_dir / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = stats_dir
    
    filepath = save_dir / f"{sanitize_filename(filename)}.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved summary: {filepath}")
    return filepath

def save_summary_statistics(percentages, avg_ale_norm_list, avg_epi_norm_list, 
                           avg_tot_norm_list, correlation_list, function_name, 
                           noise_type='heteroscedastic', func_type=''):
    """Helper function to save summary statistics and create summary plot
    
    Returns:
        tuple: (stats_df, fig) - DataFrame with statistics and matplotlib figure
    """
    # Create DataFrame with summary statistics
    stats_df = pd.DataFrame({
        'Percentage': percentages,
        'Avg_Aleatoric_norm': avg_ale_norm_list,
        'Avg_Epistemic_norm': avg_epi_norm_list,
        'Avg_Total_norm': avg_tot_norm_list,
        'Correlation_Epi_Ale': correlation_list
    })
    
    # Save statistics to CSV and Excel (Excel is saved automatically)
    save_statistics(stats_df, f"uncertainties_summary_{function_name}_{noise_type}", 
                    subfolder=f"{noise_type}/{func_type}", save_excel=True)
    
    # Create and save summary plots (uncertainties and correlations)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Normalized Average Uncertainties
    ax1.plot(percentages, avg_ale_norm_list, 'o-', linewidth=2, markersize=8, 
             label='Aleatoric Uncertainty', color='green')
    ax1.plot(percentages, avg_epi_norm_list, 's-', linewidth=2, markersize=8, 
             label='Epistemic Uncertainty', color='orange')
    ax1.plot(percentages, avg_tot_norm_list, '^-', linewidth=2, markersize=8, 
             label='Total Uncertainty', color='blue')
    ax1.set_xlabel('Training Data Percentage (%)', fontsize=12)
    ax1.set_ylabel('Normalized Average Uncertainty', fontsize=12)
    ax1.set_title(f'Normalized Average Uncertainties vs Training Data Percentage\n{function_name} Function ({noise_type.capitalize()})', 
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
    ax2.set_title(f'Correlation: Epistemic vs Aleatoric Uncertainty\n{function_name} Function ({noise_type.capitalize()})', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)
    ax2.set_ylim(-1.05, 1.05)
    
    plt.tight_layout()
    
    save_plot(fig, f"uncertainties_summary_{function_name}_{noise_type}", 
              subfolder=f"{noise_type}/{func_type}")
    
    return stats_df, fig