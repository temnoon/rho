"""
APLG Acceptance Test Routes

Provides comprehensive test endpoints to validate all APLG claim sets
and ensure compatibility with the formalized specifications. These tests
verify that each operation meets the documented mathematical and safety requirements.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import asyncio
from datetime import datetime

from routes.aplg_routes import (
    apply_channel, integrability_test, residue_analysis, consent_gate, rank_docs,
    audit_report, measure, init_rho,
    ApplyChannelRequest, IntegrabilityTestRequest, ResidueRequest, 
    ConsentGateRequest, RankDocsRequest
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/aplg-test", tags=["aplg-testing"])

# Test Suite Models
class TestResult(BaseModel):
    """Individual test result."""
    test_name: str
    claim_set: str
    status: str  # pass, fail, error
    expected: Any
    actual: Any
    error_message: Optional[str] = None
    execution_time: float

class TestSuiteResult(BaseModel):
    """Complete test suite result."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    execution_time: float
    test_results: List[TestResult]

# Test Cases for Each Claim Set
CLAIM_A_TESTS = [
    {
        "name": "basic_channel_application",
        "description": "Test basic CPTP channel application",
        "input": {
            "segment": "The weather is pleasant today.",
            "alpha": 0.3,
            "channel_type": "rank_one_update"
        },
        "expected": {
            "success": True,
            "audit.cptp_projected": True,
            "audit.trace_error_threshold": 1e-6
        }
    },
    {
        "name": "high_alpha_channel",
        "description": "Test channel application with high alpha",
        "input": {
            "segment": "This is a transformative moment of realization.",
            "alpha": 0.8,
            "channel_type": "rank_one_update"
        },
        "expected": {
            "success": True,
            "deltas.bures_step_exists": True
        }
    }
]

CLAIM_B_TESTS = [
    {
        "name": "integrable_text",
        "description": "Test integrability with simple neutral text",
        "input": {
            "text": "The sun rises in the east.",
            "tolerance": 1e-3
        },
        "expected": {
            "verdict": "compatible",
            "bures_gap_threshold": 1e-3
        }
    },
    {
        "name": "non_integrable_text",
        "description": "Test integrability with contradictory statements",
        "input": {
            "text": "Yes, definitely no, maybe absolutely certain uncertainty.",
            "tolerance": 1e-3
        },
        "expected": {
            "verdict": "incompatible",
            "bures_gap_above_threshold": True
        }
    }
]

CLAIM_C_TESTS = [
    {
        "name": "ironic_residue_detection",
        "description": "Test residue detection with ironic content",
        "input": {
            "base_text": "I can resist everything except temptation.",
            "variants": [
                "I can resist everything but temptation.",
                "Everything can be resisted by me, except temptation."
            ]
        },
        "expected": {
            "residue_norm_above": 0.01,
            "interpretation_contains": "residue"
        }
    },
    {
        "name": "neutral_residue_detection",
        "description": "Test residue detection with neutral paraphrases",
        "input": {
            "base_text": "The weather is nice today.",
            "variants": [
                "Today's weather is pleasant.",
                "It's a nice day weather-wise."
            ]
        },
        "expected": {
            "residue_norm_below": 0.01,
            "interpretation_contains": "Low residue"
        }
    }
]

CLAIM_G_TESTS = [
    {
        "name": "risky_content_detection",
        "description": "Test risk detection for manipulative content",
        "input": {
            "content": "You MUST believe this immediately! Fear what happens if you don't!",
            "alpha": 0.5,
            "user_consent_override": True
        },
        "expected": {
            "risk_assessment.risk_level": "high",
            "risk_assessment.requires_consent": True,
            "consent_status.checkpoint_created": True
        }
    },
    {
        "name": "neutral_content_approval",
        "description": "Test automatic approval for neutral content",
        "input": {
            "content": "The library has many interesting books.",
            "alpha": 0.3
        },
        "expected": {
            "risk_assessment.risk_level": "low",
            "consent_status.proceeded": True,
            "application_result": "not_null"
        }
    }
]

CLAIM_H_TESTS = [
    {
        "name": "basic_ranking",
        "description": "Test basic document ranking",
        "input": {
            "targets": "fiction,science,history",
            "k": 5
        },
        "expected": {
            "results_length": 5,
            "results_have_predicted_eig": True
        }
    }
]

CLAIM_D_TESTS = [
    {
        "name": "invariant_preserving_edit",
        "description": "Test text editing while preserving quantum invariants",
        "input": {
            "original_text": "The weather is bad today.",
            "invariant_specs": ["purity"],
            "transformation_targets": ["positive tone"],
            "max_iterations": 5
        },
        "expected": {
            "success": True,
            "edited_text_different": True,
            "final_score_above": 0.3
        }
    }
]

CLAIM_E_TESTS = [
    {
        "name": "curriculum_sequencing",
        "description": "Test optimal learning sequence planning",
        "input": {
            "content_items": ["Introduction to concepts", "Basic practice", "Advanced applications"],
            "learning_objectives": ["conceptual understanding", "practical skills"],
            "max_sequence_length": 3
        },
        "expected": {
            "sequence_length": 3,
            "optimization_score_above": 0.0
        }
    }
]

CLAIM_F_TESTS = [
    {
        "name": "bures_visualization",
        "description": "Test Bures-preserving trajectory visualization",
        "input": {
            "rho_trajectory": [],  # Will be populated with test matrices
            "visualization_type": "bures_manifold",
            "preserve_geometry": True
        },
        "expected": {
            "bures_distances_preserved": True,
            "geodesic_paths_included": True
        }
    }
]

CLAIM_I_TESTS = [
    {
        "name": "audit_retrieval",
        "description": "Test audit information retrieval",
        "input": {
            "report_id": "test_audit_id"
        },
        "expected": {
            "tolerances.trace_tol": 1e-8,
            "replay.reproducible": True
        }
    }
]

async def run_test_case(test_case: Dict[str, Any], claim_set: str, rho_id: str) -> TestResult:
    """
    Run a single test case and return the result.
    """
    start_time = datetime.now()
    
    try:
        # Execute the test based on claim set
        if claim_set == "A":
            request = ApplyChannelRequest(
                rho_id=rho_id,
                **test_case["input"]
            )
            actual = await apply_channel(request)
            
        elif claim_set == "B":
            request = IntegrabilityTestRequest(
                rho0_id=rho_id,
                **test_case["input"]
            )
            actual = await integrability_test(request)
            
        elif claim_set == "C":
            request = ResidueRequest(
                rho0_id=rho_id,
                **test_case["input"]
            )
            actual = await residue_analysis(request)
            
        elif claim_set == "D":
            from routes.aplg_routes import edit_with_invariants, InvariantEditRequest
            request = InvariantEditRequest(
                rho_id=rho_id,
                **test_case["input"]
            )
            actual = await edit_with_invariants(request)
            
        elif claim_set == "E":
            from routes.aplg_routes import plan_learning_sequence, CurriculumRequest
            request = CurriculumRequest(
                rho_id=rho_id,
                **test_case["input"]
            )
            actual = await plan_learning_sequence(request)
            
        elif claim_set == "F":
            from routes.aplg_routes import visualize_bures_trajectory, VisualizationRequest
            # Create test trajectory for visualization
            test_rho_ids = [rho_id]
            for i in range(2):
                from routes.matrix_routes import rho_init
                init_result = await init_rho(seed_text=f"Test trajectory point {i}")
                test_rho_ids.append(init_result["rho"])
            
            request = VisualizationRequest(
                rho_trajectory=test_rho_ids,
                **{k: v for k, v in test_case["input"].items() if k != "rho_trajectory"}
            )
            actual = await visualize_bures_trajectory(request)
            
        elif claim_set == "G":
            request = ConsentGateRequest(
                rho_id=rho_id,
                **test_case["input"]
            )
            actual = await consent_gate(request)
            
        elif claim_set == "H":
            actual = await rank_docs(
                rho_id=rho_id,
                **test_case["input"]
            )
            
        elif claim_set == "I":
            actual = await audit_report(test_case["input"]["report_id"])
            
        else:
            raise ValueError(f"Unknown claim set: {claim_set}")
        
        # Validate results against expectations
        expected = test_case["expected"]
        status = "pass"
        error_message = None
        
        # Check each expected condition
        for key, expected_value in expected.items():
            if "." in key:
                # Nested key like "audit.cptp_projected"
                parts = key.split(".")
                actual_value = actual
                for part in parts:
                    if isinstance(actual_value, dict) and part in actual_value:
                        actual_value = actual_value[part]
                    else:
                        actual_value = None
                        break
            else:
                actual_value = actual.get(key) if isinstance(actual, dict) else None
            
            # Validate based on expected value type and content
            if key.endswith("_above") and isinstance(expected_value, (int, float)):
                field_name = key.replace("_above", "")
                field_value = actual.get(field_name, 0)
                if not (isinstance(field_value, (int, float)) and field_value > expected_value):
                    status = "fail"
                    error_message = f"{field_name} ({field_value}) not above {expected_value}"
                    break
                    
            elif key.endswith("_below") and isinstance(expected_value, (int, float)):
                field_name = key.replace("_below", "")
                field_value = actual.get(field_name, float('inf'))
                if not (isinstance(field_value, (int, float)) and field_value < expected_value):
                    status = "fail"
                    error_message = f"{field_name} ({field_value}) not below {expected_value}"
                    break
                    
            elif key.endswith("_contains") and isinstance(expected_value, str):
                field_name = key.replace("_contains", "")
                field_value = str(actual.get(field_name, ""))
                if expected_value.lower() not in field_value.lower():
                    status = "fail"
                    error_message = f"{field_name} does not contain '{expected_value}'"
                    break
                    
            elif key.endswith("_length") and isinstance(expected_value, int):
                field_name = key.replace("_length", "")
                field_value = actual.get(field_name, [])
                if not (hasattr(field_value, '__len__') and len(field_value) == expected_value):
                    status = "fail"
                    error_message = f"{field_name} length ({len(field_value) if hasattr(field_value, '__len__') else 'N/A'}) != {expected_value}"
                    break
                    
            elif key.endswith("_threshold") and isinstance(expected_value, (int, float)):
                if not (isinstance(actual_value, (int, float)) and abs(actual_value) < expected_value):
                    status = "fail"
                    error_message = f"{key} ({actual_value}) exceeds threshold {expected_value}"
                    break
                    
            elif expected_value == "not_null":
                if actual_value is None:
                    status = "fail"
                    error_message = f"{key} is null"
                    break
                    
            elif expected_value != actual_value:
                status = "fail"
                error_message = f"{key}: expected {expected_value}, got {actual_value}"
                break
        
    except Exception as e:
        status = "error"
        actual = None
        error_message = str(e)
        expected = test_case["expected"]
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return TestResult(
        test_name=test_case["name"],
        claim_set=claim_set,
        status=status,
        expected=expected,
        actual=actual,
        error_message=error_message,
        execution_time=execution_time
    )

@router.post("/run_claim_tests/{claim_set}")
async def run_claim_tests(claim_set: str) -> TestSuiteResult:
    """
    Run all tests for a specific APLG claim set.
    """
    start_time = datetime.now()
    
    # Get test cases for the claim set
    if claim_set == "A":
        test_cases = CLAIM_A_TESTS
    elif claim_set == "B":
        test_cases = CLAIM_B_TESTS
    elif claim_set == "C":
        test_cases = CLAIM_C_TESTS
    elif claim_set == "D":
        test_cases = CLAIM_D_TESTS
    elif claim_set == "E":
        test_cases = CLAIM_E_TESTS
    elif claim_set == "F":
        test_cases = CLAIM_F_TESTS
    elif claim_set == "G":
        test_cases = CLAIM_G_TESTS
    elif claim_set == "H":
        test_cases = CLAIM_H_TESTS
    elif claim_set == "I":
        test_cases = CLAIM_I_TESTS
    else:
        raise HTTPException(status_code=400, detail=f"Unknown claim set: {claim_set}")
    
    # Create test matrix
    test_init = await init_rho(seed_text=f"Test matrix for claim set {claim_set}")
    test_rho_id = test_init["rho"]
    
    try:
        # Run all test cases
        test_results = []
        for test_case in test_cases:
            result = await run_test_case(test_case, claim_set, test_rho_id)
            test_results.append(result)
        
        # Calculate summary statistics
        passed = sum(1 for r in test_results if r.status == "pass")
        failed = sum(1 for r in test_results if r.status == "fail")
        errors = sum(1 for r in test_results if r.status == "error")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return TestSuiteResult(
            suite_name=f"APLG Claim Set {claim_set}",
            total_tests=len(test_cases),
            passed=passed,
            failed=failed,
            errors=errors,
            execution_time=execution_time,
            test_results=test_results
        )
        
    finally:
        # Clean up test matrix
        from routes.matrix_routes import STATE
        if test_rho_id in STATE:
            del STATE[test_rho_id]

@router.post("/run_all_tests")
async def run_all_aplg_tests():
    """
    Run comprehensive test suite for all implemented APLG claim sets.
    """
    start_time = datetime.now()
    
    implemented_claims = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    suite_results = []
    
    for claim_set in implemented_claims:
        try:
            result = await run_claim_tests(claim_set)
            suite_results.append(result)
        except Exception as e:
            logger.error(f"Failed to run tests for claim set {claim_set}: {e}")
            # Create error result
            error_result = TestSuiteResult(
                suite_name=f"APLG Claim Set {claim_set}",
                total_tests=0,
                passed=0,
                failed=0,
                errors=1,
                execution_time=0.0,
                test_results=[TestResult(
                    test_name="suite_execution",
                    claim_set=claim_set,
                    status="error",
                    expected="successful_execution",
                    actual=None,
                    error_message=str(e),
                    execution_time=0.0
                )]
            )
            suite_results.append(error_result)
    
    # Calculate overall statistics
    total_tests = sum(r.total_tests for r in suite_results)
    total_passed = sum(r.passed for r in suite_results)
    total_failed = sum(r.failed for r in suite_results)
    total_errors = sum(r.errors for r in suite_results)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "comprehensive_test_results": True,
        "overall_summary": {
            "total_claim_sets": len(implemented_claims),
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "execution_time": execution_time
        },
        "claim_set_results": suite_results,
        "aplg_compliance": {
            "implemented_claims": implemented_claims,
            "passing_claims": [r.suite_name for r in suite_results if r.failed == 0 and r.errors == 0],
            "failing_claims": [r.suite_name for r in suite_results if r.failed > 0 or r.errors > 0]
        }
    }

@router.get("/test_specifications")
async def get_test_specifications():
    """
    Return the complete test specifications for all claim sets.
    """
    return {
        "claim_set_A": {
            "description": "Text → CPTP Channel → ρ Update",
            "tests": CLAIM_A_TESTS,
            "requirements": [
                "CPTP channel compliance",
                "Trace preservation",
                "Positive semidefinite projection"
            ]
        },
        "claim_set_B": {
            "description": "Integrability Test",
            "tests": CLAIM_B_TESTS,
            "requirements": [
                "Path-independence detection",
                "Bures distance measurement",
                "Tolerance-based verdict"
            ]
        },
        "claim_set_C": {
            "description": "Residue/Holonomy Detection",
            "tests": CLAIM_C_TESTS,
            "requirements": [
                "Paraphrase loop analysis",
                "Irony detection",
                "Principal axis identification"
            ]
        },
        "claim_set_D": {
            "description": "Invariant-Preserving Editor",
            "tests": CLAIM_D_TESTS,
            "requirements": [
                "Quantum invariant preservation",
                "Text transformation",
                "Iterative optimization",
                "Target achievement"
            ]
        },
        "claim_set_E": {
            "description": "Curriculum/Sequencing",
            "tests": CLAIM_E_TESTS,
            "requirements": [
                "Learning objective optimization",
                "Content sequencing",
                "Readiness assessment",
                "Trajectory planning"
            ]
        },
        "claim_set_F": {
            "description": "Bures-Preserving Visualization",
            "tests": CLAIM_F_TESTS,
            "requirements": [
                "Geometric distance preservation",
                "Trajectory visualization",
                "Geodesic computation",
                "Interactive rendering"
            ]
        },
        "claim_set_G": {
            "description": "Consent/Agency Risk Gating",
            "tests": CLAIM_G_TESTS,
            "requirements": [
                "Risk assessment",
                "Consent processing",
                "Checkpoint creation",
                "Rollback capability"
            ]
        },
        "claim_set_H": {
            "description": "Reader-Aware Retrieval",
            "tests": CLAIM_H_TESTS,
            "requirements": [
                "Information gain ranking",
                "Reader state consideration",
                "Top-k results"
            ]
        },
        "claim_set_I": {
            "description": "Audit & Reproducibility",
            "tests": CLAIM_I_TESTS,
            "requirements": [
                "Complete operation logging",
                "Reproducible replay",
                "Parameter traceability"
            ]
        }
    }

@router.post("/validate_implementation")
async def validate_aplg_implementation():
    """
    Perform comprehensive validation of APLG implementation against specifications.
    """
    validation_results = []
    
    # Test each implemented claim set
    implemented_claims = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    
    for claim_set in implemented_claims:
        try:
            # Run claim set tests
            test_result = await run_claim_tests(claim_set)
            
            # Determine compliance
            is_compliant = test_result.failed == 0 and test_result.errors == 0
            
            validation_results.append({
                "claim_set": claim_set,
                "compliant": is_compliant,
                "tests_passed": test_result.passed,
                "tests_total": test_result.total_tests,
                "issues": [r.error_message for r in test_result.test_results if r.status != "pass"]
            })
            
        except Exception as e:
            validation_results.append({
                "claim_set": claim_set,
                "compliant": False,
                "tests_passed": 0,
                "tests_total": 0,
                "issues": [f"Validation failed: {str(e)}"]
            })
    
    # Overall compliance assessment
    compliant_claims = [r for r in validation_results if r["compliant"]]
    overall_compliance = len(compliant_claims) / len(implemented_claims) * 100
    
    return {
        "aplg_implementation_validation": True,
        "overall_compliance_percentage": overall_compliance,
        "compliant_claim_sets": len(compliant_claims),
        "total_claim_sets": len(implemented_claims),
        "validation_details": validation_results,
        "recommendations": [
            "Review failed tests for compliance issues",
            "Ensure all quantum operations maintain mathematical invariants",
            "Verify consent/safety mechanisms for high-risk operations",
            "Check audit trail completeness for reproducibility"
        ] if overall_compliance < 100 else [
            "APLG implementation is fully compliant with specifications",
            "All claim sets pass validation tests",
            "System ready for production use with formalized operations"
        ]
    }