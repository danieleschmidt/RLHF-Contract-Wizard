# Contributing to RLHF-Contract-Wizard

Thank you for your interest in contributing to RLHF-Contract-Wizard! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10+
- JAX 0.4.25+ with CUDA support (for development)
- Node.js 18+ (for smart contract tools)
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/RLHF-Contract-Wizard.git
   cd RLHF-Contract-Wizard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,test]"
   npm install  # For smart contract tools
   ```

4. **Run tests**
   ```bash
   pytest tests/
   npm test  # Smart contract tests
   ```

## Contribution Guidelines

### Code Style

- **Python**: Follow PEP 8, use Black formatter, type hints required
- **JavaScript/Solidity**: Use Prettier, follow Solidity style guide
- **Documentation**: Use Google-style docstrings

### Legal-Blocks Language Extensions

When contributing to the Legal-Blocks DSL:

1. **Syntax Extensions**: Propose new constraint types in `docs/adr/`
2. **Verification**: Ensure Z3 integration works with new constructs
3. **Examples**: Provide real-world usage examples
4. **Tests**: Include property-based tests for new features

### Smart Contract Security

- **Security First**: All contracts must pass security audit checklist
- **Gas Optimization**: Minimize gas costs while maintaining security
- **Upgradability**: Follow proxy patterns for contract upgrades
- **Testing**: 100% test coverage for smart contracts

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests first (TDD approach)
   - Implement the feature
   - Update documentation
   - Run full test suite

3. **Commit with conventional commits**
   ```bash
   git commit -m "feat: add stakeholder voting mechanism"
   git commit -m "fix: resolve contract compilation edge case"
   git commit -m "docs: update API reference for rewards module"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Testing Requirements

- **Unit Tests**: >95% coverage required
- **Integration Tests**: End-to-end scenarios
- **Property Tests**: For contract verification
- **Security Tests**: Smart contract vulnerability scanning

### Documentation Standards

- **API Documentation**: Automated from docstrings
- **Examples**: Working code examples for all features
- **Tutorials**: Step-by-step guides for common use cases
- **ADRs**: Architecture decisions documented in `docs/adr/`

## Areas for Contribution

### High Priority

- **Verification Engine**: Z3 optimization, Lean 4 integration
- **Performance**: JAX optimization, caching strategies
- **Security**: Smart contract auditing, formal verification
- **Documentation**: Tutorials, examples, API reference

### Research Areas

- **Constitutional AI**: Integration with constitutional training
- **Zero-Knowledge**: Private contract verification
- **Multi-Chain**: Cross-chain deployment strategies
- **Legal Framework**: Jurisdiction-specific compliance

### Community Contributions

- **Contract Templates**: Industry-specific templates
- **Tools**: CLI utilities, IDE integrations
- **Examples**: Real-world use cases and demos
- **Translations**: Documentation in multiple languages

## Review Process

1. **Automated Checks**: CI/CD pipeline must pass
2. **Code Review**: At least one maintainer approval
3. **Security Review**: For smart contract changes
4. **Documentation Review**: Technical writing team review

## Community Guidelines

- **Be Respectful**: Follow our Code of Conduct
- **Be Patient**: Reviews take time, especially for complex changes
- **Be Collaborative**: Engage with feedback constructively
- **Be Safe**: Security is paramount in this project

## Getting Help

- **Discord**: Join our [community Discord](https://discord.gg/rlhf-contracts)
- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Office Hours**: Weekly maintainer office hours (see calendar)

## Recognition

Contributors are recognized through:
- **Contributors list** in README
- **Release notes** acknowledgments
- **Conference presentations** co-authorship opportunities
- **Research collaborations** for significant contributions

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to making AI systems safer and more aligned! 🚀