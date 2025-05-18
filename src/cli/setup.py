"""Setup script for Federated Learning Framework CLI tools.

Provides installation and configuration utilities for the command-line interface tools.
"""

from setuptools import setup, find_packages

setup(
    name='fl-framework-cli',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click>=8.1.0',
        'rich>=13.0.0',
    ],
    entry_points={
        'console_scripts': [
            'fl=cli.cli_manager:cli',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='CLI tools for Federated Learning Framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/federated-learning-framework',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)