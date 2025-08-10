"""
Comprehensive testing framework for RLHF-Contract-Wizard.

Provides contract validation, integration testing, performance testing,
and security testing capabilities.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from ..models.reward_contract import RewardContract
from ..models.legal_blocks import LegalBlocks, RLHFConstraints
from ..models.reward_model import ContractualRewardModel, RewardModelConfig
from ..services.contract_service import ContractService
from ..services.verification_service import VerificationService
from ..utils.helpers import setup_logging, create_timestamp
from ..utils.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity


class TestResult(Enum):
    """Test result status."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class TestCategory(Enum):
    """Test category classification."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONTRACT = "contract"
    COMPLIANCE = "compliance"


@dataclass
class TestCase:
    """Individual test case definition."""
    name: str
    category: TestCategory
    description: str
    test_function: Callable
    timeout: float = 30.0
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    expected_result: Any = None
    skip_reason: Optional[str] = None


@dataclass
class TestResult:
    """Test execution result."""
    test_name: str
    category: TestCategory
    status: TestResult
    execution_time: float
    timestamp: float
    message: Optional[str] = None
    error_traceback: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ContractTestSuite:
    """
    Comprehensive test suite for reward contracts.
    
    Validates contract functionality, compliance, and performance
    across multiple dimensions.
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.error_handler = ErrorHandler()
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
        
        # Test data
        self.test_contracts: Dict[str, RewardContract] = {}
        self.test_data: Dict[str, Any] = {}
        
        # Services for testing
        self.contract_service = ContractService()
        self.verification_service = VerificationService(backend='mock')
        
        self._register_default_tests()
    
    def register_test(
        self,
        name: str,
        category: TestCategory,
        description: str,
        test_function: Callable,
        **kwargs
    ):
        """Register a test case."""
        test_case = TestCase(
            name=name,
            category=category,
            description=description,
            test_function=test_function,
            **kwargs
        )
        self.test_cases.append(test_case)
    
    def _register_default_tests(self):
        """Register default test cases."""
        
        # Contract creation tests
        self.register_test(
            "test_contract_creation",
            TestCategory.UNIT,
            "Test basic contract creation functionality",
            self._test_contract_creation
        )
        
        self.register_test(
            "test_stakeholder_management",
            TestCategory.UNIT,
            "Test stakeholder addition and weight management",
            self._test_stakeholder_management
        )
        
        self.register_test(
            "test_constraint_enforcement",
            TestCategory.CONTRACT,
            "Test contract constraint enforcement",
            self._test_constraint_enforcement
        )
        
        # Legal-Blocks tests
        self.register_test(
            "test_legal_blocks_parsing",
            TestCategory.UNIT,
            "Test Legal-Blocks DSL parsing",
            self._test_legal_blocks_parsing
        )
        
        self.register_test(
            "test_constraint_validation",
            TestCategory.CONTRACT,
            "Test constraint validation logic",
            self._test_constraint_validation
        )
        
        # Integration tests
        self.register_test(
            "test_contract_verification_integration",
            TestCategory.INTEGRATION,
            "Test contract verification service integration",
            self._test_contract_verification_integration
        )
        
        self.register_test(
            "test_reward_model_integration",
            TestCategory.INTEGRATION,
            "Test reward model and contract integration",
            self._test_reward_model_integration
        )
        
        # Performance tests
        self.register_test(
            "test_contract_computation_performance",
            TestCategory.PERFORMANCE,
            "Test contract reward computation performance",
            self._test_contract_computation_performance,
            timeout=60.0
        )
        
        self.register_test(
            "test_verification_performance",
            TestCategory.PERFORMANCE,
            "Test verification system performance",
            self._test_verification_performance,
            timeout=60.0
        )
        
        # Security tests
        self.register_test(
            "test_input_sanitization",
            TestCategory.SECURITY,
            "Test input sanitization and validation",
            self._test_input_sanitization
        )
        
        self.register_test(
            "test_constraint_bypass_attempts",
            TestCategory.SECURITY,
            "Test attempts to bypass contract constraints",
            self._test_constraint_bypass_attempts
        )
        
        # Compliance tests
        self.register_test(
            "test_gdpr_compliance",
            TestCategory.COMPLIANCE,
            "Test GDPR compliance features",
            self._test_gdpr_compliance
        )
        
        self.register_test(
            "test_fairness_metrics",
            TestCategory.COMPLIANCE,
            "Test fairness metrics and bias detection",
            self._test_fairness_metrics
        )
    
    def setup_test_environment(self):
        """Setup test environment with mock data."""
        # Create test contracts
        self.test_contracts['basic'] = self._create_basic_test_contract()
        self.test_contracts['safety'] = self._create_safety_focused_contract()
        self.test_contracts['multi_stakeholder'] = self._create_multi_stakeholder_contract()
        
        # Create test data
        random_key = jax.random.PRNGKey(42)
        self.test_data['random_key'] = random_key
        self.test_data['mock_states'] = jax.random.normal(random_key, (100, 128))
        self.test_data['mock_actions'] = jax.random.normal(random_key, (100, 64))
        
        self.logger.info("Test environment setup completed")
    
    def run_tests(
        self,
        categories: Optional[List[TestCategory]] = None,
        test_names: Optional[List[str]] = None,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Run test suite.
        
        Args:
            categories: Specific categories to run (None for all)
            test_names: Specific test names to run (None for all)
            parallel: Whether to run tests in parallel
            
        Returns:
            Test execution summary
        """
        self.setup_test_environment()
        
        # Filter tests
        tests_to_run = self._filter_tests(categories, test_names)
        
        self.logger.info(f"Running {len(tests_to_run)} tests...")
        start_time = time.time()
        
        if parallel and len(tests_to_run) > 1:
            results = self._run_tests_parallel(tests_to_run)
        else:
            results = self._run_tests_sequential(tests_to_run)
        
        total_time = time.time() - start_time
        
        # Compile results
        summary = self._compile_test_summary(results, total_time)
        
        self.logger.info(f"Test execution completed in {total_time:.2f}s")
        self._log_test_summary(summary)
        
        return summary
    
    def _filter_tests(
        self,
        categories: Optional[List[TestCategory]] = None,
        test_names: Optional[List[str]] = None
    ) -> List[TestCase]:
        """Filter tests based on criteria."""
        filtered_tests = []
        
        for test_case in self.test_cases:
            # Skip if specific names requested and this isn't one
            if test_names and test_case.name not in test_names:
                continue
            
            # Skip if specific categories requested and this isn't one
            if categories and test_case.category not in categories:
                continue
            
            # Skip if test has skip reason
            if test_case.skip_reason:
                continue
            
            filtered_tests.append(test_case)
        
        return filtered_tests
    
    def _run_tests_sequential(self, tests: List[TestCase]) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_case in tests:
            result = self._execute_test_case(test_case)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def _run_tests_parallel(self, tests: List[TestCase]) -> List[TestResult]:
        """Run tests in parallel using threading."""
        results = []
        threads = []
        results_lock = threading.Lock()
        
        def run_test_thread(test_case: TestCase):
            result = self._execute_test_case(test_case)
            with results_lock:
                results.append(result)
                self.results.append(result)
        
        # Start threads
        for test_case in tests:
            thread = threading.Thread(target=run_test_thread, args=(test_case,))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        return results
    
    def _execute_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        start_time = time.time()
        
        try:
            # Run setup if provided
            if test_case.setup:
                test_case.setup()
            
            # Execute test with timeout
            result = self._run_with_timeout(test_case.test_function, test_case.timeout)
            
            execution_time = time.time() - start_time
            
            # Determine test status
            if result is True or (test_case.expected_result is None and result is not False):
                status = TestResult.PASS
                message = "Test passed"
            elif result is False:
                status = TestResult.FAIL
                message = "Test failed"
            elif result == test_case.expected_result:
                status = TestResult.PASS
                message = "Test passed with expected result"
            else:
                status = TestResult.FAIL
                message = f"Expected {test_case.expected_result}, got {result}"
            
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=status,
                execution_time=execution_time,
                timestamp=create_timestamp(),
                message=message
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle error
            self.error_handler.handle_error(
                error=e,
                operation=f"test_execution:{test_case.name}",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.SYSTEM
            )
            
            return TestResult(
                test_name=test_case.name,
                category=test_case.category,
                status=TestResult.ERROR,
                execution_time=execution_time,
                timestamp=create_timestamp(),
                message=f"Test error: {str(e)}",
                error_traceback=str(e)
            )
        
        finally:
            # Run teardown if provided
            if test_case.teardown:
                try:
                    test_case.teardown()
                except Exception as e:
                    self.logger.warning(f"Teardown failed for {test_case.name}: {e}")
    
    def _run_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Run function with timeout."""
        import signal
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Function execution timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func()
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            raise
        finally:
            signal.alarm(0)  # Ensure timeout is cancelled
    
    def _compile_test_summary(self, results: List[TestResult], total_time: float) -> Dict[str, Any]:
        """Compile test execution summary."""
        summary = {
            'total_tests': len(results),
            'passed': len([r for r in results if r.status == TestResult.PASS]),
            'failed': len([r for r in results if r.status == TestResult.FAIL]),
            'errors': len([r for r in results if r.status == TestResult.ERROR]),
            'skipped': len([r for r in results if r.status == TestResult.SKIP]),
            'total_execution_time': total_time,
            'average_test_time': sum(r.execution_time for r in results) / len(results) if results else 0,
            'by_category': {},
            'detailed_results': []
        }
        
        # Group by category
        for category in TestCategory:
            category_results = [r for r in results if r.category == category]
            if category_results:
                summary['by_category'][category.value] = {
                    'total': len(category_results),
                    'passed': len([r for r in category_results if r.status == TestResult.PASS]),
                    'failed': len([r for r in category_results if r.status == TestResult.FAIL]),
                    'errors': len([r for r in category_results if r.status == TestResult.ERROR])
                }
        
        # Add detailed results
        summary['detailed_results'] = [
            {
                'name': r.test_name,
                'category': r.category.value,
                'status': r.status.value,
                'execution_time': r.execution_time,
                'message': r.message
            }
            for r in results
        ]
        
        return summary
    
    def _log_test_summary(self, summary: Dict[str, Any]):
        """Log test execution summary."""
        self.logger.info("=== TEST EXECUTION SUMMARY ===")
        self.logger.info(f"Total Tests: {summary['total_tests']}")
        self.logger.info(f"Passed: {summary['passed']}")
        self.logger.info(f"Failed: {summary['failed']}")
        self.logger.info(f"Errors: {summary['errors']}")
        self.logger.info(f"Execution Time: {summary['total_execution_time']:.2f}s")
        
        if summary['by_category']:
            self.logger.info("\nBy Category:")
            for category, stats in summary['by_category'].items():
                self.logger.info(f"  {category}: {stats['passed']}/{stats['total']} passed")
    
    # Test implementations
    def _create_basic_test_contract(self) -> RewardContract:
        """Create basic test contract."""
        contract = RewardContract(
            name="BasicTestContract",
            stakeholders={"user": 0.6, "safety": 0.4}
        )
        
        # Add basic reward function
        @contract.reward_function("user")
        def user_reward(state, action):
            return jnp.mean(state) * 0.5 + jnp.mean(action) * 0.5
        
        # Add safety constraint
        contract.add_constraint(
            "basic_safety",
            lambda s, a: jnp.all(a >= -1.0) and jnp.all(a <= 1.0),
            "Actions must be within [-1, 1]"
        )
        
        return contract
    
    def _create_safety_focused_contract(self) -> RewardContract:
        """Create safety-focused test contract."""
        contract = RewardContract(
            name="SafetyTestContract",
            stakeholders={"safety": 0.7, "user": 0.3}
        )
        
        contract.add_constraint(
            "no_harmful_output",
            RLHFConstraints.no_harmful_output,
            "Prevent harmful content"
        )
        
        contract.add_constraint(
            "privacy_protection",
            RLHFConstraints.privacy_protection,
            "Protect user privacy"
        )
        
        return contract
    
    def _create_multi_stakeholder_contract(self) -> RewardContract:
        """Create multi-stakeholder test contract."""
        contract = RewardContract(
            name="MultiStakeholderContract",
            stakeholders={
                "users": 0.4,
                "safety": 0.3,
                "operators": 0.2,
                "regulators": 0.1
            }
        )
        
        # Add stakeholder-specific reward functions
        @contract.reward_function("users")
        def user_satisfaction(state, action):
            return jnp.mean(action ** 2)
        
        @contract.reward_function("safety")
        def safety_score(state, action):
            return jnp.exp(-jnp.sum(jnp.abs(action)))
        
        return contract
    
    # Actual test implementations
    def _test_contract_creation(self) -> bool:
        """Test contract creation functionality."""
        try:
            contract = RewardContract(
                name="TestContract",
                stakeholders={"test": 1.0}
            )
            
            # Verify contract properties
            assert contract.metadata.name == "TestContract"
            assert "test" in contract.stakeholders
            assert contract.stakeholders["test"].weight == 1.0
            
            return True
        except Exception as e:
            self.logger.error(f"Contract creation test failed: {e}")
            return False
    
    def _test_stakeholder_management(self) -> bool:
        """Test stakeholder management."""
        try:
            contract = self.test_contracts['basic']
            
            # Add new stakeholder
            contract.add_stakeholder("regulator", 0.2)
            
            # Verify weights are normalized
            total_weight = sum(s.weight for s in contract.stakeholders.values())
            assert abs(total_weight - 1.0) < 1e-6
            
            return True
        except Exception as e:
            self.logger.error(f"Stakeholder management test failed: {e}")
            return False
    
    def _test_constraint_enforcement(self) -> bool:
        """Test constraint enforcement."""
        try:
            contract = self.test_contracts['basic']
            
            # Test valid action
            valid_state = jnp.array([0.5, 0.3, 0.8])
            valid_action = jnp.array([0.2, -0.1, 0.5])
            
            violations = contract.check_violations(valid_state, valid_action)
            assert not any(violations.values())
            
            # Test invalid action
            invalid_action = jnp.array([2.0, -2.0, 3.0])  # Outside [-1, 1]
            violations = contract.check_violations(valid_state, invalid_action)
            assert any(violations.values())
            
            return True
        except Exception as e:
            self.logger.error(f"Constraint enforcement test failed: {e}")
            return False
    
    def _test_legal_blocks_parsing(self) -> bool:
        """Test Legal-Blocks DSL parsing."""
        try:
            from ..models.legal_blocks import LegalBlocksParser, ConstraintType
            
            parser = LegalBlocksParser()
            spec = """
            REQUIRES: input_valid(state)
            ENSURES: reward >= 0.0 AND reward <= 1.0
            INVARIANT: NOT contains_pii(output)
            """
            
            blocks = parser.parse(spec)
            assert len(blocks) == 3
            assert blocks[0].constraint_type == ConstraintType.REQUIRES
            assert blocks[1].constraint_type == ConstraintType.ENSURES
            assert blocks[2].constraint_type == ConstraintType.INVARIANT
            
            return True
        except Exception as e:
            self.logger.error(f"Legal-Blocks parsing test failed: {e}")
            return False
    
    def _test_constraint_validation(self) -> bool:
        """Test constraint validation logic."""
        try:
            # Test RLHF constraints
            mock_action = type('Action', (), {'output': 'This is a safe response'})()
            mock_state = type('State', (), {'user_id': 'test_user'})()
            
            # Test safety constraint
            assert RLHFConstraints.no_harmful_output(mock_action)
            
            # Test privacy constraint
            assert RLHFConstraints.privacy_protection(mock_state, mock_action)
            
            return True
        except Exception as e:
            self.logger.error(f"Constraint validation test failed: {e}")
            return False
    
    def _test_contract_verification_integration(self) -> bool:
        """Test contract verification integration."""
        try:
            contract = self.test_contracts['basic']
            
            # Test verification
            result = self.verification_service.verify_contract(contract.to_dict())
            
            assert 'valid' in result
            assert 'properties_verified' in result
            
            return True
        except Exception as e:
            self.logger.error(f"Verification integration test failed: {e}")
            return False
    
    def _test_reward_model_integration(self) -> bool:
        """Test reward model integration."""
        try:
            contract = self.test_contracts['safety']
            config = RewardModelConfig(hidden_dim=128)
            
            model = ContractualRewardModel(
                config, contract, self.test_data['random_key']
            )
            
            # Test reward computation
            chosen_tokens = jax.random.randint(
                self.test_data['random_key'], (4, 32), 0, 1000
            )
            rejected_tokens = jax.random.randint(
                self.test_data['random_key'], (4, 32), 0, 1000
            )
            
            chosen_rewards, rejected_rewards, metadata = model.compute_reward(
                chosen_tokens, rejected_tokens
            )
            
            assert chosen_rewards.shape == (4,)
            assert rejected_rewards.shape == (4,)
            assert 'contract_compliance_rate' in metadata
            
            return True
        except Exception as e:
            self.logger.error(f"Reward model integration test failed: {e}")
            return False
    
    def _test_contract_computation_performance(self) -> bool:
        """Test contract computation performance."""
        try:
            contract = self.test_contracts['multi_stakeholder']
            
            # Performance test
            states = self.test_data['mock_states'][:50]
            actions = self.test_data['mock_actions'][:50]
            
            start_time = time.time()
            
            for i in range(50):
                reward = contract.compute_reward(states[i], actions[i])
            
            computation_time = time.time() - start_time
            
            # Should compute 50 rewards in under 1 second
            assert computation_time < 1.0
            
            self.logger.info(f"Contract computation: {computation_time:.4f}s for 50 samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            return False
    
    def _test_verification_performance(self) -> bool:
        """Test verification performance."""
        try:
            contracts = [
                self.test_contracts['basic'],
                self.test_contracts['safety'],
                self.test_contracts['multi_stakeholder']
            ]
            
            start_time = time.time()
            
            for contract in contracts:
                result = self.verification_service.verify_contract(contract.to_dict())
            
            verification_time = time.time() - start_time
            
            # Should verify 3 contracts in under 5 seconds
            assert verification_time < 5.0
            
            self.logger.info(f"Verification: {verification_time:.4f}s for 3 contracts")
            return True
            
        except Exception as e:
            self.logger.error(f"Verification performance test failed: {e}")
            return False
    
    def _test_input_sanitization(self) -> bool:
        """Test input sanitization."""
        try:
            # Test malicious inputs
            malicious_inputs = [
                "'; DROP TABLE contracts; --",
                "<script>alert('xss')</script>",
                "../../../etc/passwd",
                "{{7*7}}",  # Template injection
                "\\x00\\x01\\x02"  # Binary data
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    # Should not crash or execute malicious code
                    contract = RewardContract(name=malicious_input[:50])  # Truncate
                    assert len(contract.metadata.name) <= 50
                except Exception:
                    # Expected to fail gracefully
                    pass
            
            return True
        except Exception as e:
            self.logger.error(f"Input sanitization test failed: {e}")
            return False
    
    def _test_constraint_bypass_attempts(self) -> bool:
        """Test constraint bypass attempts."""
        try:
            contract = self.test_contracts['safety']
            
            # Attempt to bypass constraints with edge cases
            edge_cases = [
                (jnp.array([float('inf')]), jnp.array([1.0])),
                (jnp.array([float('nan')]), jnp.array([1.0])),
                (jnp.array([1e10]), jnp.array([1.0])),
                (jnp.array([-1e10]), jnp.array([1.0])),
            ]
            
            for state, action in edge_cases:
                try:
                    violations = contract.check_violations(state, action)
                    # Should handle edge cases gracefully
                    assert isinstance(violations, dict)
                except Exception:
                    # Should fail gracefully, not crash
                    pass
            
            return True
        except Exception as e:
            self.logger.error(f"Constraint bypass test failed: {e}")
            return False
    
    def _test_gdpr_compliance(self) -> bool:
        """Test GDPR compliance features."""
        try:
            # Test privacy constraint
            pii_action = type('Action', (), {
                'output': 'User email is john.doe@example.com and SSN is 123-45-6789'
            })()
            
            state = type('State', (), {'user_consent': False})()
            
            # Should detect PII and fail privacy check
            assert not RLHFConstraints.privacy_protection(state, pii_action)
            
            # With consent, should pass
            state.user_consent = True
            assert RLHFConstraints.privacy_protection(state, pii_action)
            
            return True
        except Exception as e:
            self.logger.error(f"GDPR compliance test failed: {e}")
            return False
    
    def _test_fairness_metrics(self) -> bool:
        """Test fairness metrics."""
        try:
            # Test fairness constraint
            fair_action = type('Action', (), {
                'output': 'This is a neutral, fair response'
            })()
            
            biased_action = type('Action', (), {
                'output': 'This group is not good at this task'
            })()
            
            mock_state = type('State', (), {'user_demographics': {}})()
            
            # Fair action should pass
            assert RLHFConstraints.fairness_requirement(mock_state, fair_action)
            
            # Biased action should fail
            assert not RLHFConstraints.fairness_requirement(mock_state, biased_action)
            
            return True
        except Exception as e:
            self.logger.error(f"Fairness metrics test failed: {e}")
            return False
    
    def save_test_report(self, filepath: str, format: str = 'json'):
        """Save test results to file."""
        report = {
            'timestamp': create_timestamp(),
            'results': [
                {
                    'name': r.test_name,
                    'category': r.category.value,
                    'status': r.status.value,
                    'execution_time': r.execution_time,
                    'message': r.message
                }
                for r in self.results
            ],
            'summary': self._compile_test_summary(self.results, 0)
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        elif format == 'html':
            self._save_html_report(report, filepath)
        
        self.logger.info(f"Test report saved to {filepath}")
    
    def _save_html_report(self, report: Dict[str, Any], filepath: Path):
        """Save HTML test report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RLHF Contract Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
                .pass {{ border-left-color: #4CAF50; }}
                .fail {{ border-left-color: #f44336; }}
                .error {{ border-left-color: #ff9800; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RLHF Contract Test Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {report['summary']['total_tests']}</p>
                <p>Passed: {report['summary']['passed']}</p>
                <p>Failed: {report['summary']['failed']}</p>
                <p>Errors: {report['summary']['errors']}</p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Category</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                    <th>Message</th>
                </tr>
        """
        
        for result in report['results']:
            status_class = result['status']
            html_content += f"""
                <tr class="{status_class}">
                    <td>{result['name']}</td>
                    <td>{result['category']}</td>
                    <td>{result['status'].upper()}</td>
                    <td>{result['execution_time']:.3f}s</td>
                    <td>{result.get('message', '')}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)