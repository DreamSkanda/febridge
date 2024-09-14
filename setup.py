"""Setup for pip package."""
import unittest
from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    'jax', 'haiku', 'optax', 'pickle',
    'os', 'sys', 're',
    'functools', 'itertools', 'typing'
]

def febridge_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('./test', pattern='test_*.py')
    return test_suite

print(find_packages())

setup(
    name='febridge',
    version='0.0.1',
    description='Free energy difference calculation between two systems.',
    url='https://github.com/DreamSkanda/febridge',
    author='Lu Zhao',
    author_email='zhaolu@iphy.ac.cn',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['pytest']},
    platforms=['any'],
    test_suite='setup.febridge_test_suite'
)
