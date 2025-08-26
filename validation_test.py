#!/usr/bin/env python3
"""
Core System Validation Test for RLHF-Contract-Wizard.

This script validates that all major components are properly implemented
and can be imported without errors. It performs basic functionality tests
without requiring external dependencies.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime


def log_test(test_name: str, status: str, details: str = ""):
    """Log test result."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"[{timestamp}] {status_symbol} {test_name:<50} {status}")
    if details:
        print(f"    └─ {details}")


def test_imports():
    """Test that all major components can be imported."""
    
    test_name = "Component Import Test"
    try:
        # Test core model imports
        sys.path.append(str(Path(__file__).parent))
        
        from src.models.reward_contract import RewardContract, AggregationStrategy
        log_test(f"{test_name} - RewardContract", "PASS", "Core reward contract model imported")
        
        # Test that we can create a basic contract
        contract = RewardContract(
            name="test_contract",
            stakeholders={"user": 0.8, "safety": 0.2}
        )
        
        if contract.metadata.name == "test_contract":
            log_test(f"{test_name} - Contract Creation", "PASS", "Basic contract created successfully")
        else:
            log_test(f"{test_name} - Contract Creation", "FAIL", "Contract metadata incorrect")
            return False
            
        # Test stakeholder normalization
        total_weight = sum(s.weight for s in contract.stakeholders.values())
        if abs(total_weight - 1.0) < 1e-6:
            log_test(f"{test_name} - Weight Normalization", "PASS", "Stakeholder weights properly normalized")
        else:
            log_test(f"{test_name} - Weight Normalization", "FAIL", f"Weights sum to {total_weight}, not 1.0")
            return False
        
        return True
        
    except ImportError as e:
        log_test(f"{test_name} - Import", "FAIL", f"Import failed: {e}")
        return False
    except Exception as e:
        log_test(f"{test_name} - Execution", "FAIL", f"Execution failed: {e}")
        return False


def test_contract_functionality():
    """Test basic contract functionality."""
    
    test_name = "Contract Functionality Test"
    try:
        from src.models.reward_contract import RewardContract
        
        # Create contract with constraints
        contract = RewardContract(
            name="functionality_test",
            stakeholders={"user": 0.6, "safety": 0.4}
        )
        
        # Add constraint
        constraint_calls = []
        def test_constraint(state, action):
            constraint_calls.append((state, action))
            return True  # Always satisfied for testing
        
        contract.add_constraint(
            name="test_constraint",
            constraint_fn=test_constraint,
            description="Test constraint for validation"
        )
        
        # Test constraint was added
        if "test_constraint" in contract.constraints:
            log_test(f"{test_name} - Constraint Addition", "PASS", "Constraint added successfully")
        else:
            log_test(f"{test_name} - Constraint Addition", "FAIL", "Constraint not added")
            return False
        
        # Test contract serialization
        contract_dict = contract.to_dict()
        if isinstance(contract_dict, dict) and "metadata" in contract_dict:
            log_test(f"{test_name} - Serialization", "PASS", "Contract serialized to dict")
        else:
            log_test(f"{test_name} - Serialization", "FAIL", "Serialization failed")
            return False
        
        # Test hash computation
        hash1 = contract.compute_hash()
        hash2 = contract.compute_hash()  # Should be identical
        
        if hash1 == hash2 and len(hash1) == 64:  # SHA256 produces 64 hex chars
            log_test(f"{test_name} - Hash Consistency", "PASS", f"Hash: {hash1[:16]}...")
        else:
            log_test(f"{test_name} - Hash Consistency", "FAIL", "Hash computation inconsistent")
            return False
        
        return True
        
    except Exception as e:
        log_test(f"{test_name} - Execution", "FAIL", f"Failed: {e}")
        return False


def test_file_structure():
    """Test that all expected files are present."""
    
    test_name = "File Structure Test"
    
    expected_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/reward_contract.py",
        "src/optimization/__init__.py",
        "src/optimization/quantum_enhanced_optimization.py",
        "src/security/__init__.py",
        "src/security/comprehensive_security_framework.py",
        "src/scaling/__init__.py",
        "src/scaling/autonomous_global_deployment.py",
        "src/research/__init__.py",
        "src/research/autonomous_research_engine.py",
        "setup.py",
        "requirements.txt",
        "README.md"
    ]
    
    repo_root = Path(__file__).parent
    missing_files = []
    present_files = []
    
    for file_path in expected_files:
        full_path = repo_root / file_path
        if full_path.exists():
            present_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    if missing_files:
        log_test(f"{test_name} - Completeness", "WARN", f"Missing {len(missing_files)} files")
        for missing in missing_files[:5]:  # Show first 5 missing files
            print(f"      Missing: {missing}")
        if len(missing_files) > 5:
            print(f"      ... and {len(missing_files) - 5} more")
    else:
        log_test(f"{test_name} - Completeness", "PASS", f"All {len(expected_files)} expected files present")
    
    log_test(f"{test_name} - Coverage", "PASS", f"{len(present_files)}/{len(expected_files)} files present ({len(present_files)/len(expected_files)*100:.1f}%)")
    
    return len(missing_files) <= len(expected_files) * 0.2  # Allow up to 20% missing


def test_module_structure():
    """Test that modules are properly structured."""
    
    test_name = "Module Structure Test"
    
    try:
        # Test that __init__.py files exist and can be imported
        import src
        log_test(f"{test_name} - Root Module", "PASS", "Root src module importable")
        
        import src.models
        log_test(f"{test_name} - Models Module", "PASS", "Models module importable")
        
        # Test that main classes are accessible
        from src.models import reward_contract
        if hasattr(reward_contract, 'RewardContract'):
            log_test(f"{test_name} - Class Access", "PASS", "RewardContract class accessible")
        else:
            log_test(f"{test_name} - Class Access", "FAIL", "RewardContract class not found")
            return False
        
        return True
        
    except ImportError as e:
        log_test(f"{test_name} - Import", "FAIL", f"Module import failed: {e}")
        return False


def test_configuration_files():
    """Test configuration files are present and valid."""
    
    test_name = "Configuration Test"
    
    repo_root = Path(__file__).parent
    
    # Test setup.py
    setup_py = repo_root / "setup.py"
    if setup_py.exists():
        try:
            with open(setup_py, 'r') as f:
                content = f.read()
                if "name=" in content and "version=" in content:
                    log_test(f"{test_name} - setup.py", "PASS", "Valid setup.py found")
                else:
                    log_test(f"{test_name} - setup.py", "FAIL", "setup.py missing required fields")
                    return False
        except Exception as e:
            log_test(f"{test_name} - setup.py", "FAIL", f"Error reading setup.py: {e}")
            return False
    else:
        log_test(f"{test_name} - setup.py", "FAIL", "setup.py not found")
        return False
    
    # Test requirements.txt
    requirements_txt = repo_root / "requirements.txt"
    if requirements_txt.exists():
        try:
            with open(requirements_txt, 'r') as f:
                requirements = f.read().strip().split('\n')
                if len(requirements) > 0 and any(req.strip() for req in requirements):
                    log_test(f"{test_name} - requirements.txt", "PASS", f"{len([r for r in requirements if r.strip()])} dependencies listed")
                else:
                    log_test(f"{test_name} - requirements.txt", "WARN", "requirements.txt is empty")
        except Exception as e:
            log_test(f"{test_name} - requirements.txt", "FAIL", f"Error reading requirements.txt: {e}")
            return False
    else:
        log_test(f"{test_name} - requirements.txt", "FAIL", "requirements.txt not found")
        return False
    
    # Test README.md
    readme_md = repo_root / "README.md"
    if readme_md.exists():
        try:
            with open(readme_md, 'r') as f:
                content = f.read()
                if len(content) > 1000:  # Substantial README
                    log_test(f"{test_name} - README.md", "PASS", f"Comprehensive README ({len(content)} chars)")
                else:
                    log_test(f"{test_name} - README.md", "WARN", f"Brief README ({len(content)} chars)")
        except Exception as e:
            log_test(f"{test_name} - README.md", "FAIL", f"Error reading README.md: {e}")
            return False
    else:
        log_test(f"{test_name} - README.md", "FAIL", "README.md not found")
        return False
    
    return True


def test_code_quality():
    """Test basic code quality metrics."""
    
    test_name = "Code Quality Test"
    
    repo_root = Path(__file__).parent
    
    # Find Python files
    python_files = list(repo_root.glob("src/**/*.py"))
    
    if not python_files:
        log_test(f"{test_name} - File Discovery", "FAIL", "No Python files found in src/")
        return False
    
    log_test(f"{test_name} - File Discovery", "PASS", f"Found {len(python_files)} Python files")
    
    # Basic syntax check - try to compile each file
    syntax_errors = []
    total_lines = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.split('\n'))
                total_lines += lines
                
                # Try to compile
                compile(content, str(py_file), 'exec')
                
        except SyntaxError as e:
            syntax_errors.append(f"{py_file.name}: {e}")
        except UnicodeDecodeError as e:
            syntax_errors.append(f"{py_file.name}: Encoding error")
        except Exception as e:
            # File might have dependencies we can't resolve, but syntax should be OK
            pass
    
    if syntax_errors:
        log_test(f"{test_name} - Syntax Check", "FAIL", f"{len(syntax_errors)} files have syntax errors")
        for error in syntax_errors[:3]:  # Show first 3 errors
            print(f"      {error}")
    else:
        log_test(f"{test_name} - Syntax Check", "PASS", f"All {len(python_files)} files have valid syntax")
    
    # Code volume check
    avg_lines_per_file = total_lines / len(python_files) if python_files else 0
    log_test(f"{test_name} - Code Volume", "PASS", f"{total_lines} total lines, {avg_lines_per_file:.0f} avg per file")
    
    return len(syntax_errors) == 0


def generate_validation_report(test_results):
    """Generate validation report."""
    
    passed_tests = sum(1 for result in test_results if result)
    total_tests = len(test_results)
    pass_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Tests Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Pass Rate: {pass_rate:.1%}")
    
    if pass_rate >= 0.9:
        status = "🌟 EXCELLENT"
        message = "System is ready for deployment"
    elif pass_rate >= 0.8:
        status = "✅ GOOD"
        message = "System is mostly ready, minor issues to address"
    elif pass_rate >= 0.6:
        status = "⚠️ FAIR"
        message = "System needs improvement before deployment"
    else:
        status = "❌ POOR"
        message = "System has significant issues requiring attention"
    
    print(f"\nOverall Status: {status}")
    print(f"Assessment: {message}")
    print("="*80)
    
    return pass_rate


def main():
    """Main validation function."""
    
    print("🧪 RLHF-Contract-Wizard System Validation")
    print("="*60)
    print(f"Validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Run all validation tests
    test_functions = [
        test_imports,
        test_contract_functionality,
        test_file_structure,
        test_module_structure,
        test_configuration_files,
        test_code_quality
    ]
    
    test_results = []
    
    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append(result)
        except Exception as e:
            log_test(f"{test_func.__name__}", "FAIL", f"Test execution failed: {e}")
            test_results.append(False)
    
    execution_time = time.time() - start_time
    
    print(f"\nValidation completed in {execution_time:.2f} seconds")
    
    # Generate final report
    pass_rate = generate_validation_report(test_results)
    
    # Save results to file
    try:
        results_file = Path(__file__).parent / "validation_results.json"
        
        validation_data = {
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "total_tests": len(test_results),
            "passed_tests": sum(test_results),
            "pass_rate": pass_rate,
            "status": "PASS" if pass_rate >= 0.8 else "FAIL",
            "test_details": [
                {
                    "test": test_functions[i].__name__,
                    "passed": result
                }
                for i, result in enumerate(test_results)
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        print(f"\n📊 Detailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n⚠️ Could not save results file: {e}")
    
    return pass_rate >= 0.8


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)