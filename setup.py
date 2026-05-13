from setuptools import setup, find_packages

setup(
    name="arbnet",
    version="0.1.0",
    description="Architecturally no-arbitrage neural option pricers with rough-volatility inductive bias",
    author="ArbNet Authors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.10",
    ],
)
