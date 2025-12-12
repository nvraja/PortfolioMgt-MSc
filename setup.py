# This is for legacy installation 
from setuptools import setup, find_packages

setup(
    name="hybrid_rl_portfolio",
    version="0.1.0",
    description="Hybrid RL + QP portfolio management",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pyarrow",
        "torch>=2.0",
        "gymnasium>=0.28",
        "osqp",
        "cvxpy",
        "tqdm",
        "ruamel.yaml",
        "seaborn",
        "plotly",
        "scikit-learn",
    ],
    include_package_data=True,
)
