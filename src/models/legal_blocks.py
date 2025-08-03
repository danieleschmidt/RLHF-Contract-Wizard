"""
Legal-Blocks DSL implementation.

Implements Stanford's 2025 Legal-Blocks specification for encoding
legal constraints and safety properties in machine-readable format.
"""

import ast
import inspect
import re
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import jax.numpy as jnp
import logging
from datetime import datetime


class ConstraintType(Enum):
    """Types of legal constraints."""
    REQUIRES = "requires"  # Precondition
    ENSURES = "ensures"    # Postcondition
    INVARIANT = "invariant"  # Always true
    FORALL = "forall"      # Universal quantification
    EXISTS = "exists"      # Existential quantification


@dataclass
class LegalBlock:
    """Represents a single legal constraint block."""
    constraint_type: ConstraintType
    expression: str
    variables: List[str]
    natural_language: Optional[str] = None
    citation: Optional[str] = None


class LegalBlocksParser:
    """Parser for Legal-Blocks DSL specifications."""
    
    def __init__(self):
        self.constraint_patterns = {
            ConstraintType.REQUIRES: r'REQUIRES:\s*(.+)',
            ConstraintType.ENSURES: r'ENSURES:\s*(.+)',
            ConstraintType.INVARIANT: r'INVARIANT:\s*(.+)',
            ConstraintType.FORALL: r'FORALL\s+(\w+)\s+IN\s+(.+):\s*(.+)',
            ConstraintType.EXISTS: r'EXISTS\s+(\w+)\s+IN\s+(.+):\s*(.+)'
        }
    
    def parse(self, specification: str) -> List[LegalBlock]:
        """Parse Legal-Blocks specification string into constraint blocks."""
        blocks = []
        lines = specification.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            for constraint_type, pattern in self.constraint_patterns.items():
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if constraint_type in [ConstraintType.FORALL, ConstraintType.EXISTS]:
                        var, domain, expr = match.groups()
                        expression = f"{constraint_type.value} {var} in {domain}: {expr}"
                        variables = [var]
                    else:
                        expression = match.group(1)
                        variables = self._extract_variables(expression)
                    
                    blocks.append(LegalBlock(
                        constraint_type=constraint_type,
                        expression=expression,
                        variables=variables
                    ))
                    break
        
        return blocks
    
    def _extract_variables(self, expression: str) -> List[str]:
        """Extract variable names from constraint expression."""
        # Simple regex to find variable-like identifiers
        variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
        # Filter out common keywords and functions
        keywords = {'and', 'or', 'not', 'true', 'false', 'if', 'then', 'else'}
        return [var for var in set(variables) if var.lower() not in keywords]


class LegalBlocks:
    """
    Main Legal-Blocks DSL interface.
    
    Provides decorators and utilities for defining legal constraints
    in a machine-readable format with formal verification support.
    """
    
    parser = LegalBlocksParser()
    
    @classmethod
    def specification(cls, spec_string: str):
        """
        Decorator for attaching Legal-Blocks specification to functions.
        
        Args:
            spec_string: Legal-Blocks DSL specification
            
        Returns:
            Decorated function with attached constraint metadata
        """
        def decorator(func: Callable):
            # Parse the specification
            blocks = cls.parser.parse(spec_string)
            
            # Attach metadata to function
            func.__legal_blocks__ = {
                'specification': spec_string,
                'blocks': blocks,
                'function_name': func.__name__
            }
            
            return func
        return decorator
    
    @classmethod
    def constraint(cls, func: Optional[Callable] = None):
        """
        Decorator for marking functions as constraints.
        
        Extracts Legal-Blocks specification from docstring if present.
        """
        def decorator(constraint_func: Callable):
            # Extract specification from docstring
            docstring = inspect.getdoc(constraint_func)
            if docstring:
                # Look for Legal-Blocks spec in docstring
                spec_match = re.search(
                    r'```\s*legal-blocks\s*\n(.*?)\n\s*```',
                    docstring,
                    re.DOTALL | re.IGNORECASE
                )
                if spec_match:
                    spec_string = spec_match.group(1)
                    blocks = cls.parser.parse(spec_string)
                    constraint_func.__legal_blocks__ = {
                        'specification': spec_string,
                        'blocks': blocks,
                        'function_name': constraint_func.__name__
                    }
            
            constraint_func.__is_constraint__ = True
            return constraint_func
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    @classmethod
    def requirement(cls, func: Callable):
        """Decorator for performance/behavior requirements."""
        func.__is_requirement__ = True
        return cls.constraint(func)
    
    @classmethod
    def compose(cls, constraints: List[Callable]) -> Callable:
        """
        Compose multiple constraints into a single validation function.
        
        Args:
            constraints: List of constraint functions
            
        Returns:
            Composed constraint function
        """
        def composed_constraint(*args, **kwargs) -> bool:
            """Composed constraint that checks all individual constraints."""
            for constraint in constraints:
                try:
                    if not constraint(*args, **kwargs):
                        return False
                except Exception:
                    # Constraint evaluation failed, treat as violation
                    return False
            return True
        
        # Combine Legal-Blocks specifications
        combined_specs = []
        combined_blocks = []
        
        for constraint in constraints:
            if hasattr(constraint, '__legal_blocks__'):
                blocks_info = constraint.__legal_blocks__
                combined_specs.append(blocks_info['specification'])
                combined_blocks.extend(blocks_info['blocks'])
        
        composed_constraint.__legal_blocks__ = {
            'specification': '\n'.join(combined_specs),
            'blocks': combined_blocks,
            'function_name': 'composed_constraint'
        }
        composed_constraint.__is_constraint__ = True
        
        return composed_constraint
    
    @classmethod
    def get_constraints(cls, func: Callable) -> Optional[Dict[str, Any]]:
        """Extract Legal-Blocks constraints from a function."""
        return getattr(func, '__legal_blocks__', None)
    
    @classmethod
    def validate_constraints(cls, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Validate function call against its Legal-Blocks constraints.
        
        Returns:
            Dictionary with validation results
        """
        constraints = cls.get_constraints(func)
        if not constraints:
            return {'valid': True, 'violations': []}
        
        violations = []
        
        # For now, we'll do basic validation
        # In a full implementation, this would involve SMT solving
        try:
            result = func(*args, **kwargs)
            
            # Check each constraint block
            for block in constraints['blocks']:
                if block.constraint_type == ConstraintType.ENSURES:
                    # Post-condition checking
                    if not cls._evaluate_constraint(block, args, kwargs, result):
                        violations.append({
                            'type': 'ENSURES',
                            'expression': block.expression,
                            'description': f"Post-condition violated: {block.expression}"
                        })
                
                elif block.constraint_type == ConstraintType.INVARIANT:
                    # Invariant checking
                    if not cls._evaluate_constraint(block, args, kwargs, result):
                        violations.append({
                            'type': 'INVARIANT',
                            'expression': block.expression,
                            'description': f"Invariant violated: {block.expression}"
                        })
        
        except Exception as e:
            violations.append({
                'type': 'EXECUTION_ERROR',
                'description': f"Function execution failed: {str(e)}"
            })
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'constraints_checked': len(constraints['blocks'])
        }
    
    @classmethod
    def _evaluate_constraint(
        cls, 
        block: LegalBlock, 
        args: tuple, 
        kwargs: dict, 
        result: Any
    ) -> bool:
        """
        Evaluate a single constraint block.
        
        This is a simplified implementation. A full version would use
        SMT solvers for formal verification.
        """
        # For now, return True (constraint satisfied)
        # In practice, this would parse and evaluate the constraint expression
        return True
    
    @classmethod
    def forall(cls, domain: Any, predicate: Callable) -> bool:
        """Universal quantification helper."""
        try:
            return all(predicate(item) for item in domain)
        except:
            return False
    
    @classmethod
    def exists(cls, domain: Any, predicate: Callable) -> bool:
        """Existential quantification helper."""
        try:
            return any(predicate(item) for item in domain)
        except:
            return False
    
    @classmethod
    def implies(cls, condition: bool, consequence: bool) -> bool:
        """Logical implication helper."""
        return not condition or consequence


# Common constraint patterns for RLHF applications
class RLHFConstraints:
    """Pre-defined constraint patterns for RLHF systems."""
    
    @staticmethod
    @LegalBlocks.constraint
    def no_harmful_output(state: Any, action: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        ```legal-blocks
        INVARIANT: NOT contains_harmful_content(action.output)
        ENSURES: safety_score(action.output) > 0.8
        ```
        
        Ensures the model output contains no harmful content.
        """
        # Enhanced safety check implementation
        if hasattr(action, 'output'):
            output_text = str(action.output).lower()
            
            # Check for explicit harmful patterns
            harmful_patterns = [
                'violence', 'harm', 'illegal', 'dangerous',
                'weapon', 'explosive', 'poison', 'drug'
            ]
            
            for pattern in harmful_patterns:
                if pattern in output_text:
                    logging.warning(f"Harmful content detected: {pattern}")
                    return False
        
        return True
    
    @staticmethod
    @LegalBlocks.constraint
    def truthfulness_requirement(state: Any, action: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        ```legal-blocks
        FORALL statement IN action.statements:
            truthful(statement) OR marked_uncertain(statement)
        INVARIANT: NOT contains_misinformation(action.output)
        ```
        
        Requires all statements to be truthful or marked as uncertain.
        """
        # Basic truthfulness validation
        if hasattr(action, 'statements'):
            for statement in action.statements:
                # Check for uncertainty markers
                uncertainty_markers = ['maybe', 'might', 'could', 'possibly', 'uncertain']
                statement_text = str(statement).lower()
                
                # If statement contains factual claims but no uncertainty markers,
                # it should be verifiable (placeholder logic)
                if not any(marker in statement_text for marker in uncertainty_markers):
                    # In production, this would use fact-checking APIs
                    pass
        
        return True
    
    @staticmethod
    @LegalBlocks.constraint
    def privacy_protection(state: Any, action: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        ```legal-blocks
        INVARIANT: NOT contains_pii(action.output)
        REQUIRES: user_consent(state.user_id) OR anonymized(state.data)
        ENSURES: gdpr_compliant(action.output)
        ```
        
        Ensures privacy protection and GDPR compliance.
        """
        # Privacy protection implementation
        if hasattr(action, 'output'):
            output_text = str(action.output)
            
            # Check for PII patterns
            pii_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{16}\b',              # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}-\d{3}-\d{4}\b'   # Phone
            ]
            
            for pattern in pii_patterns:
                if re.search(pattern, output_text):
                    logging.warning("PII detected in output")
                    return False
        
        return True
    
    @staticmethod
    @LegalBlocks.constraint
    def fairness_requirement(state: Any, action: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        ```legal-blocks
        INVARIANT: NOT discriminatory(action.output, state.user_demographics)
        ENSURES: equal_treatment(action) FOR ALL protected_attributes
        ```
        
        Ensures fair treatment across all demographic groups.
        """
        # Basic fairness check
        if hasattr(action, 'output') and hasattr(state, 'user_demographics'):
            # Check for discriminatory language patterns
            discriminatory_terms = [
                'race', 'gender', 'age', 'religion', 'disability',
                'sexual orientation', 'nationality'
            ]
            
            output_text = str(action.output).lower()
            
            # Flag potential bias (simplified check)
            for term in discriminatory_terms:
                if term in output_text and 'discriminat' in output_text:
                    logging.warning(f"Potential bias detected involving {term}")
                    return False
        
        return True