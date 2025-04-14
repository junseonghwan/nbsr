from setuptools import setup, find_packages

setup(
    name="pymirna",
    version="0.1",
    packages=find_packages(),
    install_requires=[
    'torch', 
	'numpy<2', 
	'scipy>=1.11', 
	'pandas',
    'click',
    "setuptools",
    'tables'
    ],
)
