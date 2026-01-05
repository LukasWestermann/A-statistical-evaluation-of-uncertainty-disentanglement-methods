"""
Main script to run all experiment notebooks sequentially.

This script executes all experiment notebooks in the Experiments folder
and provides a comprehensive summary report at the end.

Usage:
    python run_all_experiments.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time


def run_notebook(notebook_path):
    """
    Execute a Jupyter notebook and return success status and error message.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        tuple: (success: bool, error_message: str, execution_time: float)
    """
    print(f"\n{'='*70}")
    print(f"Running: {notebook_path.name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        # Use jupyter nbconvert to execute the notebook
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path)
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=None  # No timeout - let notebooks run as long as needed
        )
        
        execution_time = time.time() - start_time
        
        # Print any output from the notebook execution
        if result.stdout:
            print(result.stdout)
        
        print(f"\n✓ Successfully executed: {notebook_path.name}")
        print(f"  Execution time: {execution_time:.2f} seconds\n")
        return True, None, execution_time
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        error_msg = f"Timeout expired after {execution_time:.2f} seconds"
        print(f"✗ Timeout executing {notebook_path.name}: {error_msg}\n")
        return False, error_msg, execution_time
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        error_msg = e.stderr if e.stderr else str(e)
        print(f"✗ Error executing {notebook_path.name}:")
        if e.stderr:
            print(e.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout)
        print(f"\nContinuing with next notebook...\n")
        return False, error_msg, execution_time
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        print(f"✗ Unexpected error executing {notebook_path.name}: {error_msg}\n")
        print(f"Continuing with next notebook...\n")
        return False, error_msg, execution_time


def main():
    """Main execution function."""
    # Get the Experiments directory
    script_dir = Path(__file__).parent
    experiments_dir = script_dir / "Experiments"
    
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found at {experiments_dir}")
        sys.exit(1)
    
    # Define notebooks in execution order
    notebook_order = [
        "Sample Size.ipynb",
        "Undersampling.ipynb",
        "Noise_Level.ipynb",
        "OOD.ipynb",
        "OOD_Parameter_Comparison.ipynb",
        "Measurment Error.ipynb",  # Note: typo in filename preserved
    ]
    
    print("\n" + "="*70)
    print("EXPERIMENT NOTEBOOK RUNNER")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments directory: {experiments_dir}")
    print(f"Total notebooks to execute: {len(notebook_order)}")
    print("="*70)
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Execute each notebook
    for notebook_name in notebook_order:
        notebook_path = experiments_dir / notebook_name
        
        if not notebook_path.exists():
            print(f"\n⚠ Warning: {notebook_name} not found, skipping...")
            results[notebook_name] = {
                'success': False,
                'error': 'Notebook file not found',
                'execution_time': 0.0
            }
            continue
        
        success, error_msg, exec_time = run_notebook(notebook_path)
        results[notebook_name] = {
            'success': success,
            'error': error_msg,
            'execution_time': exec_time
        }
    
    total_execution_time = time.time() - total_start_time
    
    # Print summary report
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
    print("\nNotebook Results:")
    print("-"*70)
    
    successful = []
    failed = []
    
    for notebook_name, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        time_str = f"{result['execution_time']:.2f}s"
        
        print(f"{notebook_name:45} {status:12} ({time_str:>8})")
        
        if result['success']:
            successful.append(notebook_name)
        else:
            failed.append(notebook_name)
            if result['error']:
                print(f"  Error: {result['error'][:100]}...")  # Truncate long errors
    
    print("-"*70)
    print(f"\nSummary:")
    print(f"  Successful: {len(successful)}/{len(notebook_order)}")
    print(f"  Failed:     {len(failed)}/{len(notebook_order)}")
    
    if failed:
        print(f"\nFailed notebooks:")
        for nb in failed:
            print(f"  - {nb}")
    
    print("="*70 + "\n")
    
    # Exit with error code if any notebook failed
    if failed:
        sys.exit(1)
    else:
        print("All notebooks executed successfully! ✓\n")
        sys.exit(0)


if __name__ == "__main__":
    main()

