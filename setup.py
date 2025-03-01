from setuptools import setup, find_packages

setup(
    name='pyFinRisk',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'math',
        'numpy',
        'scipy',
        'scikit-learn',
        'yfinance',
    ],
    author='tzabcoder',
    description='Python financial risk management library.',
    url='https://github.com/tzabcoder/pyFinRisk',
)