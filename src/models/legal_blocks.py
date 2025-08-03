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
        Evaluate a single constraint block with basic expression evaluation.
        
        This is a simplified implementation. A full version would use
        SMT solvers for formal verification.
        """
        try:
            # Create evaluation context
            context = {
                'args': args,
                'kwargs': kwargs,
                'result': result,
                'len': len,
                'sum': sum,
                'min': min,
                'max': max,
                'all': all,
                'any': any
            }
            
            # Add arguments by name if possible
            if hasattr(args, '__iter__') and len(args) > 0:
                context.update({
                    'state': args[0] if len(args) > 0 else None,
                    'action': args[1] if len(args) > 1 else None
                })
            
            # Simple expression evaluation for basic constraints
            expression = block.expression.lower()
            
            # Handle common constraint patterns
            if 'reward' in expression and '>=' in expression:
                # Extract numeric constraint like "reward >= 0.0"
                if hasattr(result, '__iter__') and len(result) > 0:
                    reward_value = float(result[0]) if isinstance(result[0], (int, float)) else 0.0
                else:
                    reward_value = float(result) if isinstance(result, (int, float)) else 0.0
                
                if '>= 0' in expression:
                    return reward_value >= 0.0
                elif '<= 1' in expression:
                    return reward_value <= 1.0
            
            # Handle NOT expressions
            if expression.startswith('not '):
                sub_expr = expression[4:].strip()
                if 'contains_pii' in sub_expr:
                    # Mock PII detection - in practice would use NLP models
                    return True  # Assume no PII for now
                elif 'harmful' in sub_expr:
                    # Mock harm detection
                    return True  # Assume no harm for now
            
            # Default to satisfied constraint
            return True
            
        except Exception as e:
            # If evaluation fails, consider constraint violated
            return False
    
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


# Evaluation helpers for constraint checking
class ConstraintEvaluator:
    """Helper class for evaluating Legal-Blocks constraints."""
    
    @staticmethod
    def contains_pii(text: str) -> bool:
        """Check if text contains personally identifiable information."""
        # Simplified PII detection - would use ML models in practice
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
        ]
        
        text = str(text)
        return any(re.search(pattern, text) for pattern in pii_patterns)
    
    @staticmethod
    def contains_harmful_content(text: str) -> bool:
        """Check if text contains harmful content."""
        # Simplified harm detection - would use specialized models
        harmful_keywords = [
            'violence', 'hate', 'discrimination', 'illegal', 'harmful',
            'dangerous', 'weapon', 'drug', 'suicide', 'self-harm'
        ]
        
        text = str(text).lower()
        return any(keyword in text for keyword in harmful_keywords)
    
    @staticmethod
    def safety_score(text: str) -> float:
        """Compute safety score for text (0.0 = unsafe, 1.0 = safe)."""
        # Simplified scoring - would use trained safety models
        if ConstraintEvaluator.contains_harmful_content(text):
            return 0.3
        elif ConstraintEvaluator.contains_pii(text):
            return 0.6
        else:
            return 0.9
    
    @staticmethod
    def is_truthful(statement: str) -> bool:
        """Check if statement is truthful."""
        # Mock implementation - would use fact-checking models
        uncertainty_markers = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'i think']
        statement_lower = statement.lower()
        return any(marker in statement_lower for marker in uncertainty_markers) or len(statement) < 200


# Common constraint patterns for RLHF applications
class RLHFConstraints:
    """Pre-defined constraint patterns for RLHF systems."""
    
    @staticmethod
    @LegalBlocks.constraint
    def no_harmful_output(action: Any) -> bool:
        """
        ```legal-blocks
        INVARIANT: NOT contains_harmful_content(action.output)
        ENSURES: safety_score(action.output) > 0.8
        ```
        
        Ensures the model output contains no harmful content.
        """
        try:
            output_text = getattr(action, 'output', str(action))
            if ConstraintEvaluator.contains_harmful_content(output_text):
                return False
            return ConstraintEvaluator.safety_score(output_text) > 0.8
        except:
            return False
    
    @staticmethod
    @LegalBlocks.constraint
    def truthfulness_requirement(action: Any) -> bool:
        """
        ```legal-blocks
        FORALL statement IN action.statements:
            truthful(statement) OR marked_uncertain(statement)
        INVARIANT: NOT contains_misinformation(action.output)
        ```
        
        Requires all statements to be truthful or marked as uncertain.
        """
        try:
            output_text = getattr(action, 'output', str(action))
            statements = getattr(action, 'statements', [output_text])
            
            # Check each statement for truthfulness
            for statement in statements:
                if not ConstraintEvaluator.is_truthful(str(statement)):
                    return False
            
            # Check for obvious misinformation patterns
            misinformation_indicators = ['fake news', 'conspiracy', 'hoax', 'debunked']
            return not any(indicator in output_text.lower() for indicator in misinformation_indicators)
        except:
            return False
    
    @staticmethod
    @LegalBlocks.constraint
    def privacy_protection(state: Any, action: Any) -> bool:
        """
        ```legal-blocks
        INVARIANT: NOT contains_pii(action.output)
        REQUIRES: user_consent(state.user_id) OR anonymized(state.data)
        ENSURES: gdpr_compliant(action.output)
        ```
        
        Ensures privacy protection and GDPR compliance.
        """
        try:
            output_text = getattr(action, 'output', str(action))
            
            # Check for PII in output
            if ConstraintEvaluator.contains_pii(output_text):
                return False
            
            # Mock user consent check
            user_id = getattr(state, 'user_id', None)
            has_consent = getattr(state, 'user_consent', True)  # Default to True for demo
            
            # Mock data anonymization check
            is_anonymized = getattr(state, 'anonymized', True)  # Default to True for demo
            
            return has_consent or is_anonymized
        except:
            return False
    
    @staticmethod
    @LegalBlocks.constraint
    def fairness_requirement(state: Any, action: Any) -> bool:
        """
        ```legal-blocks
        INVARIANT: NOT discriminatory(action.output, state.user_demographics)
        ENSURES: equal_treatment(action) FOR ALL protected_attributes
        ```
        
        Ensures fair treatment across all demographic groups.
        """
        try:
            output_text = getattr(action, 'output', str(action))
            
            # Check for discriminatory language
            discriminatory_terms = [
                'race', 'gender', 'religion', 'nationality', 'age',
                'disability', 'sexual orientation', 'bias', 'prejudice'
            ]
            
            # Simple check - in practice would use bias detection models
            text_lower = output_text.lower()
            
            # Flag potentially discriminatory content
            for term in discriminatory_terms:
                if term in text_lower and any(neg in text_lower for neg in ['not', 'never', 'wrong', 'bad']):
                    # Contains discriminatory language in negative context
                    return False
            
            return True
        except:
            return False
    
    @staticmethod
    @LegalBlocks.constraint
    def reward_bounds(state: jnp.ndarray, action: jnp.ndarray, reward: float) -> bool:
        """
        ```legal-blocks
        REQUIRES: is_valid_state(state) AND is_valid_action(action)
        ENSURES: reward >= -1.0 AND reward <= 1.0
        INVARIANT: finite(reward)
        ```
        
        Ensures reward values are bounded and finite.
        """
        try:
            # Check if reward is finite
            if not jnp.isfinite(reward):
                return False
            
            # Check bounds
            if reward < -1.0 or reward > 1.0:
                return False
            
            return True
        except:
            return False
    
    @staticmethod
    @LegalBlocks.constraint
    def computational_efficiency(state: jnp.ndarray, action: jnp.ndarray, execution_time: float) -> bool:
        """
        ```legal-blocks
        REQUIRES: execution_time > 0
        ENSURES: execution_time < 100.0
        INVARIANT: reasonable_resource_usage(computation)
        ```
        
        Ensures computational efficiency constraints are met.
        """
        try:
            # Check execution time bounds (in milliseconds)
            if execution_time <= 0 or execution_time >= 100.0:
                return False
            
            # Mock resource usage check
            state_size = jnp.size(state) if hasattr(state, '__len__') else 1
            action_size = jnp.size(action) if hasattr(action, '__len__') else 1
            
            # Reasonable size limits
            if state_size > 10000 or action_size > 1000:
                return False
            
            return True
        except:
            return False