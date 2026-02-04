"""
Main script to run all experiment notebooks sequentially.

This script executes all experiment notebooks in the Experiments folder
and provides a comprehensive summary report at the end.

Usage:
    python run_all_experiments.py
"""

import sys
from pathlib import Path
from datetime import datetime
import time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def run_notebook(notebook_path):
    """
    Execute a Jupyter notebook cell-by-cell and return success status and error messages.
    Continues execution even if individual cells fail.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        tuple: (success: bool, error_message: str, execution_time: float, failed_cells: list)
               where failed_cells is a list of dicts with 'index', 'cell_type', and 'error' keys
    """
    print(f"\n{'='*70}")
    print(f"Running: {notebook_path.name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    failed_cells = []
    
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Track code cells for reporting
        code_cells = [(i, cell) for i, cell in enumerate(nb.cells) if cell.cell_type == 'code']
        print(f"Found {len(code_cells)} code cells to execute\n")
        
        # Execute notebook with allow_errors=True to continue on cell failures
        client = NotebookClient(nb, timeout=None, kernel_name='python3', allow_errors=True)
        
        # Execute all cells (will continue even if some fail)
        try:
            client.execute()
        except Exception as e:
            # Even if execution raises an exception, check which cells failed
            print(f"Warning: Execution raised exception: {str(e)}")
        
        # After execution, check which cells failed by examining outputs
        for cell_idx, (nb_idx, cell) in enumerate(code_cells):
            if cell.cell_type == 'code':
                # Check if cell has error output
                if hasattr(cell, 'outputs') and cell.outputs:
                    for output in cell.outputs:
                        if output.output_type == 'error':
                            error_name = output.get('ename', 'Error')
                            error_value = output.get('evalue', 'Unknown error')
                            error_msg = f"{error_name}: {error_value}"
                            
                            # Add traceback if available
                            if 'traceback' in output and output.traceback:
                                tb_lines = output.traceback[-3:] if len(output.traceback) > 3 else output.traceback
                                error_msg += f"\n  {' '.join(tb_lines)}"
                            
                            failed_cells.append({
                                'index': cell_idx,
                                'cell_type': cell.cell_type,
                                'error': error_msg
                            })
                            print(f"✗ Cell {cell_idx + 1} failed: {error_msg[:150]}...")
                            break
                    else:
                        # No error found, cell succeeded
                        if cell_idx < len(code_cells) - 1 or len(code_cells) <= 10:
                            print(f"✓ Cell {cell_idx + 1} executed successfully")
                else:
                    # No outputs yet, might be pending
                    if cell_idx < len(code_cells) - 1 or len(code_cells) <= 10:
                        print(f"✓ Cell {cell_idx + 1} executed")
        
        if len(code_cells) > 10:
            print(f"... (executed {len(code_cells)} cells total)")
        
        execution_time = time.time() - start_time
        
        # Save the notebook with execution results (even if some cells failed)
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        # Determine overall success
        all_successful = len(failed_cells) == 0
        error_message = None
        
        if failed_cells:
            error_message = f"{len(failed_cells)} cell(s) failed"
            print(f"\n⚠ Completed with {len(failed_cells)} failed cell(s)")
        else:
            print(f"\n✓ Successfully executed all cells in: {notebook_path.name}")
        
        print(f"  Execution time: {execution_time:.2f} seconds\n")
        return all_successful, error_message, execution_time, failed_cells
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        print(f"✗ Error loading/executing {notebook_path.name}: {error_msg}\n")
        print(f"Continuing with next notebook...\n")
        return False, error_msg, execution_time, failed_cells


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
        "OOD_Parameter_Comparison.ipynb",  # Note: typo in filename preserved
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
                'execution_time': 0.0,
                'failed_cells': []
            }
            continue
        
        success, error_msg, exec_time, failed_cells = run_notebook(notebook_path)
        results[notebook_name] = {
            'success': success,
            'error': error_msg,
            'execution_time': exec_time,
            'failed_cells': failed_cells
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
                print(f"  Error: {result['error']}")
            
            # Show cell-level failures if any
            if result.get('failed_cells'):
                failed_cells = result['failed_cells']
                print(f"  Failed cells: {len(failed_cells)}")
                for cell_failure in failed_cells[:5]:  # Show first 5 failed cells
                    print(f"    - Cell {cell_failure['index'] + 1}: {cell_failure['error'][:150]}...")
                if len(failed_cells) > 5:
                    print(f"    ... and {len(failed_cells) - 5} more cell(s)")
    
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

