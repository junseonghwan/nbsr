from setuptools import setup, find_packages

setup(
    name="nbsr",
    version="0.2",
    description="Negative Binomial Softmax Regression: a Bayesian compositional model for sequencing counts",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2",
        "numpy",
        "scipy>=1.11",
        "pandas",
        "click",
        "tables",
        "joblib",
        "tqdm",
        "pydeseq2",
        "patsy",
        "anndata",
    ],
    entry_points={"console_scripts": ["nbsr=nbsr.main:cli"]},
)
