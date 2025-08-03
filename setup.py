"""
Setup configuration for RLHF-Contract-Wizard.

Installs the package in development mode with all dependencies.
"""

from setuptools import setup, find_packages
import os

# Read version from package
VERSION = "0.1.0"

# Read long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, "r") as f:
        return [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#") and not line.startswith("-r")
        ]

# Base requirements
install_requires = read_requirements("requirements.txt")

# Development requirements  
dev_requires = read_requirements("requirements-dev.txt")

# Optional extras
extras_require = {
    "dev": dev_requires,
    "test": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0", 
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "httpx>=0.24.0"
    ],
    "verification": [
        "z3-solver>=4.12.0",  # SMT solver
        # "lean4>=4.0.0",  # Theorem prover (install separately)
    ],
    "blockchain": [
        "web3>=6.11.0",
        "eth-account>=0.9.0",
        "brownie>=1.19.0"
    ],
    "docs": [
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.2.0",
        "mkdocs-mermaid2-plugin>=1.1.0"
    ],
    "monitoring": [
        "prometheus-client>=0.17.0",
        "structlog>=23.1.0",
        "sentry-sdk>=1.30.0"
    ],
    "all": dev_requires  # Install everything
}

setup(
    name="rlhf-contract-wizard",
    version=VERSION,
    description="JAX library for encoding RLHF reward functions as legally-binding smart contracts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Schmidt",
    author_email="contact@rlhf-contracts.org",
    url="https://github.com/danieleschmidt/RLHF-Contract-Wizard",
    project_urls={
        "Documentation": "https://rlhf-contracts.org/docs",
        "Source Code": "https://github.com/danieleschmidt/RLHF-Contract-Wizard",
        "Bug Tracker": "https://github.com/danieleschmidt/RLHF-Contract-Wizard/issues",
        "Discussions": "https://github.com/danieleschmidt/RLHF-Contract-Wizard/discussions"
    },
    packages=find_packages(exclude=["tests*", "docs*", "scripts*"]),
    include_package_data=True,
    package_data={
        "rlhf_contract": [
            "src/database/schema.sql",
            "src/database/migrations/*.sql",
            "src/database/seeds/*.sql"
        ]
    },
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "rlhf-contract=src.cli.main:main",
            "rlhf-api=src.api.main:main"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security :: Cryptography",
        "Topic :: Office/Business :: Financial",
        "Typing :: Typed"
    ],
    keywords=[
        "rlhf", "ai-alignment", "smart-contracts", "legal-blocks", 
        "blockchain", "jax", "reinforcement-learning", "contract-verification",
        "formal-methods", "safety", "governance"
    ],
    zip_safe=False,
    platforms=["any"],
    license="Apache License 2.0"
)