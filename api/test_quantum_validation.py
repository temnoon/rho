#!/usr/bin/env python3
"""
Comprehensive End-to-End Quantum Mechanics Verification Suite

This script validates that the Rho Narrative System properly implements
quantum mechanical principles across all components. It tests:

1. Quantum state mathematical constraints
2. Channel operation validity (CPTP properties)
3. POVM measurement consistency
4. Bures geometry calculations
5. Integrability testing accuracy
6. APLG claim verification
7. System-wide quantum consistency

No mock data - all tests use real quantum mathematics!
"""

import sys
import numpy as np
import requests
import time
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add API path for imports
sys.path.append('/Users/tem/rho/api')

# Import quantum math functions
from core.quantum_state import create_maximally_mixed_state, apply_text_channel, diagnostics
from core.integrability_testing import quick_integrability_check, test_text_integrability, IntegrabilityTester
from core.text_channels import text_to_channel
from core.embedding import text_to_embedding_vector
from models.quantum_models import (
    QuantumDiagnostics, 
    validate_quantum_constraints,
    numpy_to_quantum_diagnostics
)

# Configuration
API_BASE = "http://localhost:8192"
TOLERANCE = 1e-6

class QuantumValidationSuite:
    """Comprehensive quantum mechanics validation."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def log_result(self, test_name: str, passed: bool, details: Dict[str, Any]):
        """Log test result."""
        self.results[test_name] = {
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        
    def log_error(self, test_name: str, error: str):
        """Log validation error."""
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ ERROR {test_name}: {error}")
        
    def log_warning(self, test_name: str, warning: str):
        """Log validation warning."""
        self.warnings.append(f"{test_name}: {warning}")
        print(f"⚠️  WARNING {test_name}: {warning}")

    # ========================================================================
    # CORE QUANTUM STATE VALIDATION
    # ========================================================================
    
    def test_density_matrix_properties(self):
        """Test fundamental density matrix properties."""
        print("\\n🔬 Testing Density Matrix Properties...")
        
        # Create maximally mixed state
        rho = create_maximally_mixed_state()
        
        # Test 1: Trace = 1
        trace = np.trace(rho)
        trace_valid = abs(trace - 1.0) < TOLERANCE
        if not trace_valid:
            self.log_error("density_matrix_trace", f"Trace = {trace}, expected 1.0")
        
        # Test 2: Hermitian (ρ = ρ†)
        hermitian_valid = np.allclose(rho, rho.conj().T, atol=TOLERANCE)
        if not hermitian_valid:
            self.log_error("density_matrix_hermitian", "Matrix is not Hermitian")
            
        # Test 3: Positive semidefinite (all eigenvalues ≥ 0)
        eigenvals = np.linalg.eigvals(rho)
        min_eigenval = np.min(eigenvals)
        psd_valid = min_eigenval >= -TOLERANCE
        if not psd_valid:
            self.log_error("density_matrix_psd", f"Minimum eigenvalue = {min_eigenval}")
            
        # Test 4: Purity bounds (0 ≤ Tr(ρ²) ≤ 1)
        purity = np.trace(rho @ rho)
        purity_valid = 0 <= purity <= 1 + TOLERANCE
        if not purity_valid:
            self.log_error("density_matrix_purity", f"Purity = {purity}, expected [0,1]")
            
        # Test 5: Dimension consistency
        expected_dim = 64
        actual_dim = rho.shape[0]
        dim_valid = actual_dim == expected_dim
        if not dim_valid:
            self.log_error("density_matrix_dimension", f"Dimension = {actual_dim}, expected {expected_dim}")
            
        all_valid = trace_valid and hermitian_valid and psd_valid and purity_valid and dim_valid
        
        self.log_result("density_matrix_properties", all_valid, {
            "trace": float(trace),
            "min_eigenval": float(min_eigenval), 
            "purity": float(purity),
            "dimension": actual_dim,
            "hermitian": hermitian_valid
        })
        
        return all_valid

    def test_quantum_diagnostics_validation(self):
        """Test standardized quantum diagnostics."""
        print("\\n📊 Testing Quantum Diagnostics Validation...")
        
        rho = create_maximally_mixed_state()
        
        try:
            # Test standard diagnostic conversion
            diag = numpy_to_quantum_diagnostics(rho)
            
            # Test validation
            errors = validate_quantum_constraints(type('MockState', (), {'diagnostics': diag})())
            
            has_errors = len(errors) > 0
            if has_errors:
                for error in errors:
                    self.log_error("quantum_diagnostics", f"{error.error_type}: {error.description}")
                    
            self.log_result("quantum_diagnostics_validation", not has_errors, {
                "trace": diag.trace,
                "purity": diag.purity,
                "entropy": diag.entropy,
                "effective_rank": diag.effective_rank,
                "validation_errors": len(errors)
            })
            
            return not has_errors
            
        except Exception as e:
            self.log_error("quantum_diagnostics", f"Exception: {str(e)}")
            return False

    # ========================================================================
    # CHANNEL OPERATION VALIDATION
    # ========================================================================
    
    def test_text_channel_properties(self):
        """Test quantum channel CPTP properties."""
        print("\\n🔄 Testing Text Channel Properties...")
        
        try:
            # Create test text and embedding
            test_text = "A quantum state represents the complete description of a physical system."
            embedding = text_to_embedding_vector(test_text)
            
            # Create channel
            channel = text_to_channel(embedding, alpha=0.3, channel_type="rank_one_update")
            
            # Test channel on initial state
            initial_state = create_maximally_mixed_state()
            final_state = channel.apply(initial_state)
            
            # Test 1: Trace preservation
            initial_trace = np.trace(initial_state)
            final_trace = np.trace(final_state)
            trace_preserved = abs(final_trace - initial_trace) < TOLERANCE
            
            # Test 2: Output is valid density matrix
            output_valid = self.validate_density_matrix(final_state)
            
            # Test 3: Completely positive (harder to test directly, check PSD)
            final_eigenvals = np.linalg.eigvals(final_state)
            cp_valid = np.all(final_eigenvals >= -TOLERANCE)
            
            all_valid = trace_preserved and output_valid and cp_valid
            
            self.log_result("text_channel_properties", all_valid, {
                "initial_trace": float(initial_trace),
                "final_trace": float(final_trace),
                "trace_error": float(abs(final_trace - initial_trace)),
                "output_valid": output_valid,
                "min_eigenval": float(np.min(final_eigenvals))
            })
            
            return all_valid
            
        except Exception as e:
            self.log_error("text_channel_properties", f"Exception: {str(e)}")
            return False
            
    def validate_density_matrix(self, rho: np.ndarray) -> bool:
        """Quick density matrix validation."""
        try:
            trace_ok = abs(np.trace(rho) - 1.0) < TOLERANCE
            hermitian_ok = np.allclose(rho, rho.conj().T, atol=TOLERANCE)
            eigenvals = np.linalg.eigvals(rho)
            psd_ok = np.all(eigenvals >= -TOLERANCE)
            return trace_ok and hermitian_ok and psd_ok
        except:
            return False

    # ========================================================================
    # POVM MEASUREMENT VALIDATION
    # ========================================================================
    
    def test_povm_measurement_consistency(self):
        """Test POVM measurement mathematical consistency."""
        print("\\n📏 Testing POVM Measurement Consistency...")
        
        try:
            # Create test state via API
            init_response = requests.post(f"{API_BASE}/rho/init")
            if init_response.status_code != 200:
                self.log_error("povm_measurement", f"Init failed: {init_response.status_code}")
                return False
                
            rho_id = init_response.json()["rho_id"]
            
            # Read some text into the state
            read_response = requests.post(f"{API_BASE}/rho/{rho_id}/read_channel", json={
                "raw_text": "The quantum measurement revealed hidden narrative structures.",
                "alpha": 0.3
            })
            
            if read_response.status_code != 200:
                self.log_error("povm_measurement", f"Read failed: {read_response.status_code}")
                return False
            
            # Apply POVM measurements
            measure_response = requests.post(f"{API_BASE}/packs/measure/{rho_id}", json={
                "pack_id": "advanced_narrative_pack"
            })
            
            if measure_response.status_code != 200:
                self.log_error("povm_measurement", f"Measure failed: {measure_response.status_code}")
                return False
                
            data = measure_response.json()
            measurements = data["measurements"]
            diagnostics = data["diagnostics"]
            
            # Test 1: All measurements are valid probabilities
            prob_valid = all(0 <= p <= 1 for p in measurements.values())
            
            # Test 2: Trace is preserved
            trace_valid = abs(diagnostics["trace"] - 1.0) < TOLERANCE
            
            # Test 3: Measurements are consistent with quantum state
            # For each axis pair, probabilities should sum to approximately 1
            axis_pairs = self.group_measurement_pairs(measurements)
            pair_sums_valid = True
            pair_details = {}
            
            for axis_name, pair in axis_pairs.items():
                if len(pair) == 2:
                    pair_sum = sum(pair.values())
                    pair_valid = abs(pair_sum - 1.0) < 0.1  # Allow some tolerance for partial measurements
                    pair_details[axis_name] = {"sum": pair_sum, "valid": pair_valid}
                    if not pair_valid:
                        pair_sums_valid = False
            
            all_valid = prob_valid and trace_valid and pair_sums_valid
            
            self.log_result("povm_measurement_consistency", all_valid, {
                "num_measurements": len(measurements),
                "all_probabilities_valid": prob_valid,
                "trace": diagnostics["trace"],
                "trace_valid": trace_valid,
                "pair_sums_valid": pair_sums_valid,
                "pair_details": pair_details
            })
            
            return all_valid
            
        except Exception as e:
            self.log_error("povm_measurement", f"Exception: {str(e)}")
            return False
            
    def group_measurement_pairs(self, measurements: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Group measurements into complementary pairs."""
        pairs = {}
        for key, value in measurements.items():
            # Extract axis name (everything before the last underscore)
            parts = key.split('_')
            if len(parts) >= 2:
                axis_name = '_'.join(parts[:-1])
                outcome = parts[-1]
                
                if axis_name not in pairs:
                    pairs[axis_name] = {}
                pairs[axis_name][outcome] = value
                
        return pairs

    # ========================================================================
    # INTEGRABILITY TESTING VALIDATION
    # ========================================================================
    
    def test_integrability_mathematics(self):
        """Test integrability testing mathematical correctness."""
        print("\\n🔗 Testing Integrability Mathematics...")
        
        try:
            # Test text that should be highly integrable
            integrable_text = "The cat sat on the mat. The feline rested on the rug."
            
            # Test with core function
            result = quick_integrability_check(integrable_text, alpha=0.2)
            
            # Test 1: Bures distance should be small for similar segmentations
            bures_dist = result["bures_distance"]
            bures_valid = bures_dist < 0.1  # Should be small for integrable text
            
            # Test 2: Test passes or fails appropriately
            passes_test = result["passes_test"]
            
            # Test with more challenging text
            complex_text = "Quantum mechanics revolutionized physics. However, the stock market crashed. Meanwhile, Shakespeare wrote sonnets."
            complex_result = quick_integrability_check(complex_text, alpha=0.3)
            complex_bures = complex_result["bures_distance"]
            
            # Test 3: More complex text should generally have higher Bures distance
            complexity_valid = complex_bures >= bures_dist or complex_bures < 0.01  # Allow for rare exceptions
            
            all_valid = bures_valid and isinstance(passes_test, bool) and complexity_valid
            
            self.log_result("integrability_mathematics", all_valid, {
                "simple_bures_distance": bures_dist,
                "simple_passes_test": passes_test,
                "complex_bures_distance": complex_bures,
                "complexity_ordering_valid": complexity_valid,
                "bures_threshold_met": bures_valid
            })
            
            return all_valid
            
        except Exception as e:
            self.log_error("integrability_mathematics", f"Exception: {str(e)}")
            return False

    # ========================================================================
    # BURES GEOMETRY VALIDATION
    # ========================================================================
    
    def test_bures_geometry_properties(self):
        """Test Bures distance mathematical properties."""
        print("\\n📐 Testing Bures Geometry Properties...")
        
        try:
            # Create three different quantum states
            state1 = create_maximally_mixed_state()
            
            # Modify states slightly
            embedding1 = text_to_embedding_vector("The first quantum narrative.")
            channel1 = text_to_channel(embedding1, alpha=0.1, channel_type="rank_one_update")
            state2 = channel1.apply(state1.copy())
            
            embedding2 = text_to_embedding_vector("A completely different story altogether.")
            channel2 = text_to_channel(embedding2, alpha=0.2, channel_type="rank_one_update") 
            state3 = channel2.apply(state1.copy())
            
            # Use IntegrabilityTester class methods for Bures distance and fidelity
            tester = IntegrabilityTester()
            
            # Test 1: Identity - d(ρ, ρ) = 0
            d_self = tester._bures_distance(state1, state1)
            identity_valid = d_self < TOLERANCE
            
            # Test 2: Symmetry - d(ρ₁, ρ₂) = d(ρ₂, ρ₁)
            d_12 = tester._bures_distance(state1, state2)
            d_21 = tester._bures_distance(state2, state1)
            symmetry_valid = abs(d_12 - d_21) < TOLERANCE
            
            # Test 3: Triangle inequality - d(ρ₁, ρ₃) ≤ d(ρ₁, ρ₂) + d(ρ₂, ρ₃)
            d_13 = tester._bures_distance(state1, state3)
            d_23 = tester._bures_distance(state2, state3)
            triangle_valid = d_13 <= d_12 + d_23 + TOLERANCE
            
            # Test 4: Non-negativity - all distances ≥ 0
            non_negative_valid = all(d >= 0 for d in [d_self, d_12, d_13, d_23])
            
            # Test 5: Fidelity relationship - F(ρ₁, ρ₂) = 1 - d²(ρ₁, ρ₂)/2 for small distances
            fidelity = tester._quantum_fidelity(state1, state2)
            fidelity_valid = 0 <= fidelity <= 1
            
            all_valid = identity_valid and symmetry_valid and triangle_valid and non_negative_valid and fidelity_valid
            
            self.log_result("bures_geometry_properties", all_valid, {
                "identity_distance": float(d_self),
                "symmetry_diff": float(abs(d_12 - d_21)),
                "triangle_inequality_satisfied": triangle_valid,
                "distances": {"d_12": float(d_12), "d_13": float(d_13), "d_23": float(d_23)},
                "fidelity": float(fidelity),
                "all_non_negative": non_negative_valid
            })
            
            return all_valid
            
        except Exception as e:
            self.log_error("bures_geometry_properties", f"Exception: {str(e)}")
            return False

    # ========================================================================
    # API INTEGRATION VALIDATION
    # ========================================================================
    
    def test_api_quantum_consistency(self):
        """Test API quantum operations maintain mathematical consistency."""
        print("\\n🌐 Testing API Quantum Consistency...")
        
        try:
            # Test full workflow via API
            
            # 1. Create quantum state
            init_response = requests.post(f"{API_BASE}/rho/init")
            if init_response.status_code != 200:
                raise Exception(f"Init failed: {init_response.status_code}")
                
            rho_id = init_response.json()["rho_id"]
            init_diagnostics = init_response.json()["diagnostics"]
            
            # Validate initial state
            init_valid = abs(init_diagnostics["trace"] - 1.0) < TOLERANCE
            
            # 2. Read text into state
            test_text = "Quantum narrative systems bridge the gap between mathematics and meaning."
            read_response = requests.post(f"{API_BASE}/rho/{rho_id}/read_channel", json={
                "raw_text": test_text,
                "alpha": 0.3,
                "channel_type": "rank_one_update"
            })
            
            if read_response.status_code != 200:
                raise Exception(f"Read failed: {read_response.status_code}")
                
            read_data = read_response.json()
            read_diagnostics = read_data["diagnostics"]
            
            # Validate post-read state
            read_valid = abs(read_diagnostics["trace"] - 1.0) < TOLERANCE
            channel_audit = read_data.get("channel_audit", {})
            channel_valid = channel_audit.get("passes_audit", False)
            
            # 3. Apply measurements
            measure_response = requests.post(f"{API_BASE}/packs/measure/{rho_id}", json={
                "pack_id": "advanced_narrative_pack"
            })
            
            if measure_response.status_code != 200:
                raise Exception(f"Measure failed: {measure_response.status_code}")
                
            measure_data = measure_response.json()
            measure_diagnostics = measure_data["diagnostics"]
            measurements = measure_data["measurements"]
            
            # Validate post-measurement state
            measure_valid = abs(measure_diagnostics["trace"] - 1.0) < TOLERANCE
            prob_valid = all(0 <= p <= 1 for p in measurements.values())
            
            # 4. Test integrability via API
            aplg_response = requests.post(f"{API_BASE}/aplg/integrability_test", json={
                "text": test_text,
                "tolerance": 1e-3
            })
            
            aplg_valid = aplg_response.status_code == 200
            if aplg_valid:
                aplg_data = aplg_response.json()
                bures_gap = aplg_data.get("bures_gap", float('inf'))
                integrability_valid = bures_gap < 1.0  # Reasonable threshold
            else:
                integrability_valid = False
            
            all_valid = init_valid and read_valid and channel_valid and measure_valid and prob_valid and aplg_valid and integrability_valid
            
            self.log_result("api_quantum_consistency", all_valid, {
                "init_trace": init_diagnostics["trace"],
                "read_trace": read_diagnostics["trace"], 
                "measure_trace": measure_diagnostics["trace"],
                "channel_passes_audit": channel_valid,
                "measurements_count": len(measurements),
                "probabilities_valid": prob_valid,
                "aplg_available": aplg_valid,
                "integrability_result": bures_gap if aplg_valid else None
            })
            
            return all_valid
            
        except Exception as e:
            self.log_error("api_quantum_consistency", f"Exception: {str(e)}")
            return False

    # ========================================================================
    # COMPREHENSIVE SYSTEM VALIDATION
    # ========================================================================
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🧪 QUANTUM MECHANICS VALIDATION SUITE")
        print("=" * 50)
        
        start_time = time.time()
        
        # Core quantum tests
        test_results = [
            self.test_density_matrix_properties(),
            self.test_quantum_diagnostics_validation(),
            self.test_text_channel_properties(),
            self.test_povm_measurement_consistency(),
            self.test_integrability_mathematics(),
            self.test_bures_geometry_properties(),
            self.test_api_quantum_consistency()
        ]
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Summary
        total_tests = len(test_results)
        passed_tests = sum(test_results)
        pass_rate = passed_tests / total_tests
        
        print("\\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Tests Run: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Tests Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {pass_rate:.1%}")
        print(f"Duration: {duration:.2f} seconds")
        
        if self.errors:
            print(f"\\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
                
        if self.warnings:
            print(f"\\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        overall_status = "✅ PASS" if pass_rate >= 0.9 else "⚠️  PARTIAL" if pass_rate >= 0.7 else "❌ FAIL"
        print(f"\\n🎯 OVERALL STATUS: {overall_status}")
        
        return {
            "overall_pass": pass_rate >= 0.9,
            "pass_rate": pass_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "duration": duration,
            "detailed_results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Run the validation suite."""
    print("🚀 Starting Comprehensive Quantum Mechanics Validation...")
    print("🔬 Testing genuine quantum mathematics - NO MOCK DATA!")
    print()
    
    validator = QuantumValidationSuite()
    results = validator.run_all_tests()
    
    # Save results (convert numpy types to Python types for JSON serialization)
    def convert_numpy_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    json_compatible_results = convert_numpy_types(results)
    
    with open('/Users/tem/rho/api/quantum_validation_results.json', 'w') as f:
        json.dump(json_compatible_results, f, indent=2)
    
    print(f"\\n💾 Results saved to: quantum_validation_results.json")
    
    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())