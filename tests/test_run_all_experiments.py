"""
Tests for the run_all_experiments.py script.

Tests verify that the script can:
- Find the Experiments directory
- Identify notebooks
- Execute notebooks (with mocked execution for speed)
- Handle errors gracefully
"""
import pytest
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the script functions
# Since run_all_experiments.py is a script, we import it as a module
import importlib.util
spec = importlib.util.spec_from_file_location("run_all_experiments", project_root / "run_all_experiments.py")
run_all_experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_all_experiments)


class TestRunAllExperiments:
    """Tests for run_all_experiments.py script."""
    
    def test_script_imports(self):
        """Test that the script can be imported without errors."""
        # Verify the module was loaded
        assert run_all_experiments is not None
        assert hasattr(run_all_experiments, 'run_notebook')
        assert hasattr(run_all_experiments, 'main')
        assert callable(run_all_experiments.run_notebook)
        assert callable(run_all_experiments.main)
    
    def test_experiments_directory_exists(self):
        """Test that the Experiments directory exists."""
        experiments_dir = project_root / "Experiments"
        assert experiments_dir.exists(), f"Experiments directory not found at {experiments_dir}"
        assert experiments_dir.is_dir()
    
    def test_finds_notebooks(self):
        """Test that the script can find experiment notebooks."""
        experiments_dir = project_root / "Experiments"
        
        expected_notebooks = [
            "Sample Size.ipynb",
            "Undersampling.ipynb",
            "Noise_Level.ipynb",
            "OOD.ipynb",
            "OOD_Parameter_Comparison.ipynb",
            "Measurment Error.ipynb",
        ]
        
        found_notebooks = []
        for notebook_name in expected_notebooks:
            notebook_path = experiments_dir / notebook_name
            if notebook_path.exists():
                found_notebooks.append(notebook_name)
        
        # At least some notebooks should exist
        assert len(found_notebooks) > 0, "No experiment notebooks found"
        print(f"Found {len(found_notebooks)}/{len(expected_notebooks)} expected notebooks")
    
    def test_run_notebook_success(self):
        """Test run_notebook function with successful execution (mocked)."""
        # Create a temporary notebook file
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_path = Path(tmpdir) / "test_notebook.ipynb"
            
            # Create a minimal valid notebook
            notebook_content = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "source": ["print('Hello, World!')"]
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            import json
            notebook_path.write_text(json.dumps(notebook_content))
            
            # Mock subprocess.run to simulate successful execution
            with patch('run_all_experiments.subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "Notebook executed successfully"
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                success, error_msg, exec_time = run_all_experiments.run_notebook(notebook_path)
                
                assert success is True
                assert error_msg is None
                assert exec_time >= 0
                mock_run.assert_called_once()
    
    def test_run_notebook_failure(self):
        """Test run_notebook function with failed execution (mocked)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_path = Path(tmpdir) / "test_notebook.ipynb"
            notebook_path.write_text("dummy content")
            
            # Mock subprocess.run to simulate failed execution
            with patch('run_all_experiments.subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    returncode=1,
                    cmd=["jupyter", "nbconvert"],
                    stderr="Error executing notebook"
                )
                
                success, error_msg, exec_time = run_all_experiments.run_notebook(notebook_path)
                
                assert success is False
                assert error_msg is not None
                assert "Error executing notebook" in error_msg or len(error_msg) > 0
                assert exec_time >= 0
    
    def test_run_notebook_missing_file(self):
        """Test run_notebook function with missing notebook file."""
        missing_path = Path("/nonexistent/path/test.ipynb")
        
        # Mock subprocess.run to simulate file not found
        with patch('run_all_experiments.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("File not found")
            
            success, error_msg, exec_time = run_all_experiments.run_notebook(missing_path)
            
            assert success is False
            assert error_msg is not None
            assert exec_time >= 0
    
    def test_main_function_structure(self):
        """Test that main function has correct structure (without actually running notebooks)."""
        experiments_dir = project_root / "Experiments"
        
        # Verify experiments directory exists
        assert experiments_dir.exists()
        
        # Verify notebook order is defined
        # We can't easily test the full main() without running notebooks,
        # but we can verify the structure
        notebook_order = [
            "Sample Size.ipynb",
            "Undersampling.ipynb",
            "Noise_Level.ipynb",
            "OOD.ipynb",
            "OOD_Parameter_Comparison.ipynb",
            "Measurment Error.ipynb",
        ]
        
        # Check that at least some notebooks exist
        found_count = sum(
            1 for nb in notebook_order 
            if (experiments_dir / nb).exists()
        )
        assert found_count > 0, "No notebooks found in expected order"
    
    def test_notebook_order_completeness(self):
        """Test that all expected notebooks are in the execution order."""
        # This test verifies the notebook order matches what's expected
        # We check the source code structure
        script_path = project_root / "run_all_experiments.py"
        script_content = script_path.read_text()
        
        expected_notebooks = [
            "Sample Size.ipynb",
            "Undersampling.ipynb",
            "Noise_Level.ipynb",
            "OOD.ipynb",
            "OOD_Parameter_Comparison.ipynb",
            "Measurment Error.ipynb",
        ]
        
        for notebook in expected_notebooks:
            assert notebook in script_content, f"Notebook {notebook} not found in script"
    
    @pytest.mark.skipif(
        not shutil.which("jupyter"),
        reason="jupyter command not available"
    )
    def test_jupyter_available(self):
        """Test that jupyter nbconvert is available (if jupyter is installed)."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--help"],
                capture_output=True,
                timeout=5
            )
            assert result.returncode == 0 or "nbconvert" in result.stdout.decode() or "nbconvert" in result.stderr.decode()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("jupyter nbconvert not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

