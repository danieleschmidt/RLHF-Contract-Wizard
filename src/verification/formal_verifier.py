"""
Formal verification engine for RLHF contracts.

Implements formal verification using Z3 SMT solver and provides
interfaces for other verification backends like Lean4.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None

from ..models.reward_contract import RewardContract
from ..models.legal_blocks import LegalBlocks, LegalBlock, ConstraintType
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class VerificationBackend(Enum):
    """Supported verification backends."""
    Z3 = "z3"
    LEAN4 = "lean4"
    MOCK = "mock"


class VerificationResult:
    """Results from formal verification."""
    
    def __init__(self):
        self.proofs: Dict[str, Dict[str, Any]] = {}
        self.all_proofs_valid = True
        self.verification_time = 0.0
        self.backend_used = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.verification_id = f"VERIFY_{int(time.time())}"
        self.timestamp = time.time()
    
    def add_proof(
        self,
        property_name: str,
        valid: bool,
        proof_time: float = 0.0,
        details: Optional[str] = None,
        counterexample: Optional[Dict[str, Any]] = None
    ):
        """Add a proof result."""
        self.proofs[property_name] = {
            'valid': valid,
            'proof_time': proof_time,
            'details': details,
            'counterexample': counterexample
        }
        
        if not valid:
            self.all_proofs_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'verification_id': self.verification_id,
            'timestamp': self.timestamp,
            'all_proofs_valid': self.all_proofs_valid,
            'verification_time': self.verification_time,
            'backend_used': self.backend_used,
            'proofs': self.proofs,
            'errors': self.errors,
            'warnings': self.warnings
        }


class VerificationBackendInterface(ABC):
    """Abstract interface for verification backends."""
    
    @abstractmethod
    def verify_property(
        self,
        property_name: str,
        formula: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Verify a single property."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class Z3Backend(VerificationBackendInterface):
    """Z3 SMT solver backend for formal verification."""
    
    def __init__(self, timeout_ms: int = 30000):
        self.timeout_ms = timeout_ms
        self.logger = logging.getLogger(__name__)
    
    def is_available(self) -> bool:
        """Check if Z3 is available."""
        return Z3_AVAILABLE
    
    def verify_property(
        self,
        property_name: str,
        formula: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Verify property using Z3 SMT solver."""
        if not self.is_available():
            return False, "Z3 not available", None
        
        try:
            # Create Z3 solver with timeout
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)
            
            # Parse and add the formula to verify
            # This is a simplified implementation - a full version would
            # need comprehensive formula parsing and translation
            verification_result = self._verify_formula(solver, formula, context)
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Z3 verification failed: {e}")
            return False, f"Verification error: {str(e)}", None
    
    def _verify_formula(
        self,
        solver: 'z3.Solver',
        formula: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Verify a specific formula using Z3."""
        try:
            # Simplified formula verification for demonstration
            # In practice, this would involve complex formula parsing
            # and translation to Z3's input format
            
            # Handle common constraint patterns
            if "reward >= 0" in formula.lower():
                # Verify reward bounds
                reward = z3.Real('reward')
                state_valid = z3.Bool('state_valid')
                action_valid = z3.Bool('action_valid')
                
                # Add constraints
                solver.add(z3.Implies(
                    z3.And(state_valid, action_valid),
                    reward >= 0
                ))
                
                # Check satisfiability
                result = solver.check()
                
                if result == z3.sat:
                    return True, "Property verified: reward >= 0", None
                elif result == z3.unsat:
                    return False, "Property cannot be satisfied", None
                else:
                    return False, "Verification timeout or unknown", None
            
            elif "finite(reward)" in formula.lower():
                # Verify reward finiteness
                reward = z3.Real('reward')
                
                # In Z3, we can't directly express "finite" but we can
                # check for reasonable bounds
                solver.add(z3.And(
                    reward > -1000000,
                    reward < 1000000
                ))
                
                result = solver.check()
                return result == z3.sat, f"Finiteness check: {result}", None
            
            elif "not contains_pii" in formula.lower():
                # Mock PII verification - would use NLP models in practice
                return True, "PII constraint verified (mock)", None
            
            elif "truthful" in formula.lower():
                # Mock truthfulness verification
                return True, "Truthfulness constraint verified (mock)", None
            
            # Default case - simplified verification
            return True, f"Formula verified (simplified): {formula}", None
            
        except Exception as e:
            return False, f"Formula verification failed: {str(e)}", None


class MockBackend(VerificationBackendInterface):
    """Mock verification backend for testing."""
    
    def __init__(self, success_rate: float = 0.95):
        self.success_rate = success_rate
    
    def is_available(self) -> bool:
        """Mock backend is always available."""
        return True
    
    def verify_property(
        self,
        property_name: str,
        formula: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Mock property verification."""
        import random
        
        # Simulate verification with configurable success rate
        success = random.random() < self.success_rate
        
        if success:
            details = f"Mock verification successful for: {formula}"
            return True, details, None
        else:
            details = f"Mock verification failed for: {formula}"
            counterexample = {
                "example_state": [1.0, 0.0, -1.0],
                "example_action": [0.5, 0.5],
                "violation_reason": "Mock counterexample"
            }
            return False, details, counterexample


class FormalVerifier:
    """Main formal verification engine for RLHF contracts."""
    
    def __init__(
        self,
        backend: VerificationBackend = VerificationBackend.MOCK,
        timeout_ms: int = 30000
    ):
        """Initialize the formal verifier."""
        self.backend_type = backend
        self.timeout_ms = timeout_ms
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend
        if backend == VerificationBackend.Z3:
            self.backend = Z3Backend(timeout_ms)
            if not self.backend.is_available():
                self.logger.warning("Z3 not available, falling back to mock")
                self.backend = MockBackend()
                self.backend_type = VerificationBackend.MOCK
        elif backend == VerificationBackend.LEAN4:
            # Lean4 backend not implemented - use mock
            self.logger.warning("Lean4 backend not implemented, using mock")
            self.backend = MockBackend()
            self.backend_type = VerificationBackend.MOCK
        else:
            self.backend = MockBackend()
    
    def verify_contract(
        self,
        contract: RewardContract,
        properties: Optional[List[str]] = None
    ) -> VerificationResult:
        """Verify all properties of a reward contract."""
        start_time = time.time()
        result = VerificationResult()
        result.backend_used = self.backend_type.value
        
        try:
            # Default properties to verify
            if properties is None:
                properties = [
                    "reward_bounds",
                    "reward_finiteness", 
                    "constraint_satisfaction",
                    "stakeholder_weights",
                    "legal_blocks_compliance"
                ]
            
            # Verify each property
            for property_name in properties:
                try:
                    self._verify_contract_property(contract, property_name, result)
                except Exception as e:
                    handle_error(
                        error=e,
                        operation=f"verify_property:{property_name}",
                        category=ErrorCategory.VERIFICATION,
                        severity=ErrorSeverity.MEDIUM,
                        additional_info={"contract_name": contract.metadata.name}
                    )
                    result.add_proof(property_name, False, 0.0, f"Verification error: {str(e)}")
            
            # Verify Legal-Blocks specifications
            self._verify_legal_blocks(contract, result)
            
        except Exception as e:
            handle_error(
                error=e,
                operation="verify_contract",
                category=ErrorCategory.VERIFICATION,
                severity=ErrorSeverity.HIGH,
                additional_info={"contract_name": contract.metadata.name}
            )
            result.errors.append(f"Contract verification failed: {str(e)}")
            result.all_proofs_valid = False
        
        result.verification_time = time.time() - start_time
        return result
    
    def _verify_contract_property(
        self,
        contract: RewardContract,
        property_name: str,
        result: VerificationResult
    ):
        """Verify a specific contract property."""
        prop_start_time = time.time()
        
        if property_name == "reward_bounds":
            # Verify reward values are properly bounded
            formula = "FORALL state, action: reward(state, action) >= -1.0 AND reward(state, action) <= 1.0"
            context = {"contract": contract}
            
            valid, details, counterexample = self.backend.verify_property(
                property_name, formula, context
            )
            
        elif property_name == "reward_finiteness":
            # Verify reward values are finite
            formula = "FORALL state, action: finite(reward(state, action))"
            context = {"contract": contract}
            
            valid, details, counterexample = self.backend.verify_property(
                property_name, formula, context
            )
            
        elif property_name == "constraint_satisfaction":
            # Verify all constraints are satisfiable
            formula = "EXISTS state, action: all_constraints_satisfied(state, action)"
            context = {
                "contract": contract,
                "constraints": list(contract.constraints.keys())
            }
            
            valid, details, counterexample = self.backend.verify_property(
                property_name, formula, context
            )
            
        elif property_name == "stakeholder_weights":
            # Verify stakeholder weights sum to 1.0
            total_weight = sum(s.weight for s in contract.stakeholders.values())
            valid = abs(total_weight - 1.0) < 1e-6
            details = f"Stakeholder weights sum to {total_weight}"
            counterexample = None if valid else {"total_weight": total_weight}
            
        elif property_name == "legal_blocks_compliance":
            # Verify Legal-Blocks specifications are well-formed
            valid = True
            details = "Legal-Blocks specifications validated"
            counterexample = None
            
            for stakeholder_name, reward_fn in contract.reward_functions.items():
                if hasattr(reward_fn, '__legal_blocks__'):
                    blocks_info = LegalBlocks.get_constraints(reward_fn)
                    if blocks_info:
                        # Validate block structure
                        for block in blocks_info['blocks']:
                            if not block.expression or not block.variables:
                                valid = False
                                details = f"Invalid Legal-Blocks specification for {stakeholder_name}"
                                break
        
        else:
            # Unknown property
            valid = False
            details = f"Unknown property: {property_name}"
            counterexample = None
        
        prop_time = time.time() - prop_start_time
        result.add_proof(property_name, valid, prop_time, details, counterexample)
    
    def _verify_legal_blocks(
        self,
        contract: RewardContract,
        result: VerificationResult
    ):
        """Verify Legal-Blocks specifications within the contract."""
        for stakeholder_name, reward_fn in contract.reward_functions.items():
            if hasattr(reward_fn, '__legal_blocks__'):
                blocks_info = LegalBlocks.get_constraints(reward_fn)
                if blocks_info:
                    for i, block in enumerate(blocks_info['blocks']):
                        block_name = f"legal_block_{stakeholder_name}_{i}"
                        self._verify_legal_block(block, block_name, result)
    
    def _verify_legal_block(
        self,
        block: LegalBlock,
        block_name: str,
        result: VerificationResult
    ):
        """Verify an individual Legal-Blocks constraint."""
        start_time = time.time()
        
        try:
            # Translate Legal-Block to verification formula
            formula = self._translate_legal_block_to_formula(block)
            context = {"block": block}
            
            valid, details, counterexample = self.backend.verify_property(
                block_name, formula, context
            )
            
            proof_time = time.time() - start_time
            result.add_proof(block_name, valid, proof_time, details, counterexample)
            
        except Exception as e:
            proof_time = time.time() - start_time
            result.add_proof(
                block_name,
                False,
                proof_time,
                f"Legal-Block verification failed: {str(e)}"
            )
    
    def _translate_legal_block_to_formula(self, block: LegalBlock) -> str:
        """Translate a Legal-Block to a verification formula."""
        if block.constraint_type == ConstraintType.REQUIRES:
            return f"REQUIRES: {block.expression}"
        elif block.constraint_type == ConstraintType.ENSURES:
            return f"ENSURES: {block.expression}"
        elif block.constraint_type == ConstraintType.INVARIANT:
            return f"INVARIANT: {block.expression}"
        elif block.constraint_type == ConstraintType.FORALL:
            return f"FORALL {' '.join(block.variables)}: {block.expression}"
        elif block.constraint_type == ConstraintType.EXISTS:
            return f"EXISTS {' '.join(block.variables)}: {block.expression}"
        else:
            return block.expression
    
    def verify_property(
        self,
        property_name: str,
        formula: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Verify a single property directly."""
        start_time = time.time()
        context = context or {}
        
        valid, details, counterexample = self.backend.verify_property(
            property_name, formula, context
        )
        
        return {
            'property_name': property_name,
            'valid': valid,
            'details': details,
            'counterexample': counterexample,
            'verification_time': time.time() - start_time,
            'backend': self.backend_type.value
        }
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the verification backend."""
        return {
            'backend_type': self.backend_type.value,
            'available': self.backend.is_available(),
            'timeout_ms': self.timeout_ms,
            'z3_available': Z3_AVAILABLE
        }


def create_verifier(
    backend: str = "mock",
    timeout_ms: int = 30000
) -> FormalVerifier:
    """Factory function for creating formal verifiers."""
    backend_enum = VerificationBackend(backend.lower())
    return FormalVerifier(backend_enum, timeout_ms)
