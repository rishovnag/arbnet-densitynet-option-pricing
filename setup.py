from setuptools import setup, find_packages

setup(
    name="arbnet",
    version="0.1.0",
    description=(
        "Architecturally no-arbitrage neural option pricers with "
        "rough-volatility inductive bias"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="ArbNet Authors",
    license="MIT",
    url="https://github.com/your-org/arbnet",
    packages=find_packages(include=["arbnet", "arbnet.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.10",
    ],
    extras_require={
        # Reading the bundled 91-day T-bill auctions .xlsx.
        "data": ["openpyxl>=3.1"],
        # Re-fetching raw NSE / market data via scripts/prepare_data.py.
        "fetch": ["requests>=2.28", "yfinance>=0.2", "tqdm>=4.64"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
