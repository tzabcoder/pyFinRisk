from setuptools import setup, find_packages

setup(
    name='pyFinRisk',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'math',
        'matplotlib',
        'numpy',
        'scipy',
        'statsmodels',
        'yfinance',
    ],
    author='tzabcoder',
    description='Python financial risk management library.',
    url='https://github.com/tzabcoder/pyFinRisk',
)