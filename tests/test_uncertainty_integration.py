"""
Integration tests for uncertainty decomposition.
Tests that variance and entropy methods work together correctly.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Models.MC_Dropout import mc_dropout_predict
from utils.entropy_uncertainty import entropy_uncertainty_analytical


class TestVarianceEntropyConsistency:
    """Test that variance and entropy decompositions are consistent."""
    
    def test_both_methods_from_same_raw_arrays(self, trained_mc_dropout_model):
        """Test that both variance and entropy can be computed from same raw arrays."""
        x = np.random.randn(10, 1).astype(np.float32)
        
        # Get raw arrays
        result = mc_dropout_predict(trained_mc_dropout_model, x, M=5, return_raw_arrays=True)
        mu_pred_var, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy from same raw arrays
        entropy_result = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        
        # Both should have same number of data points
        assert len(mu_pred_var) == len(entropy_result['aleatoric'])
        
        # Both should be finite
        assert np.all(np.isfinite(ale_var))
        assert np.all(np.isfinite(entropy_result['aleatoric']))
        assert np.all(np.isfinite(epi_var))
        assert np.all(np.isfinite(entropy_result['epistemic']))
        
        # Entropy should increase with variance (monotonic relationship)
        # Higher variance -> higher entropy
        correlation_ale = np.corrcoef(ale_var, entropy_result['aleatoric'])[0, 1]
        correlation_epi = np.corrcoef(epi_var, entropy_result['epistemic'])[0, 1]
        
        # Positive correlation expected
        assert correlation_ale > 0.3  # Allow some tolerance
        assert correlation_epi > 0.3  # Allow some tolerance
    
    def test_shapes_match(self, trained_mc_dropout_model):
        """Test that variance and entropy outputs have matching shapes."""
        x = np.random.randn(15, 1).astype(np.float32)
        
        result = mc_dropout_predict(trained_mc_dropout_model, x, M=5, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        entropy_result = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        
        # All should have same length
        assert len(ale_var) == len(entropy_result['aleatoric'])
        assert len(epi_var) == len(entropy_result['epistemic'])
        assert len(tot_var) == len(entropy_result['total'])
        assert len(mu_pred) == len(entropy_result['aleatoric'])


class TestEndToEndUncertainty:
    """Test full workflow from training to uncertainty computation."""
    
    def test_full_workflow_mc_dropout(self, trained_mc_dropout_model):
        """Test complete workflow: predict → variance → entropy."""
        x = np.random.randn(10, 1).astype(np.float32)
        
        # Step 1: Predict with raw arrays
        result = mc_dropout_predict(trained_mc_dropout_model, x, M=5, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Step 2: Compute entropy from raw arrays
        entropy_result = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        
        # Step 3: Verify all results are valid
        assert np.all(np.isfinite(mu_pred))
        assert np.all(np.isfinite(ale_var))
        assert np.all(np.isfinite(epi_var))
        assert np.all(np.isfinite(tot_var))
        assert np.all(np.isfinite(entropy_result['aleatoric']))
        assert np.all(np.isfinite(entropy_result['epistemic']))
        assert np.all(np.isfinite(entropy_result['total']))
        
        # Step 4: Verify relationships
        assert np.allclose(tot_var, ale_var + epi_var, rtol=1e-5)
        
        # Entropy total should be approximately aleatoric + epistemic
        entropy_sum = entropy_result['aleatoric'] + entropy_result['epistemic']
        diff = np.abs(entropy_result['total'] - entropy_sum)
        assert np.all(diff < 0.2 * np.abs(entropy_result['total']))
    
    def test_workflow_with_deep_ensemble(self, trained_deep_ensemble):
        """Test workflow with Deep Ensemble."""
        from Models.Deep_Ensemble import ensemble_predict_deep
        
        x = np.random.randn(10, 1).astype(np.float32)
        
        # Predict with raw arrays
        result = ensemble_predict_deep(trained_deep_ensemble, x, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy
        entropy_result = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        
        # Verify all valid
        assert np.all(np.isfinite(mu_pred))
        assert np.all(np.isfinite(entropy_result['aleatoric']))
        assert np.all(np.isfinite(entropy_result['epistemic']))
        assert np.all(np.isfinite(entropy_result['total']))

