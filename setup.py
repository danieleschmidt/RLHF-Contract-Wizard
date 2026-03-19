from setuptools import setup, find_packages

setup(
    name="rlhf-contract-wizard",
    version="0.1.0",
    description="Encode RLHF reward functions as auditable model card contracts",
    author="Daniel Schmidt",
    url="https://github.com/danieleschmidt/RLHF-Contract-Wizard",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
