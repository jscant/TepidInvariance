from setuptools import setup

setup(
    name='tepid_invariance',
    version='0.0.1',
    description='SE(3)-equivariant neural networks for hotspot prediction.',
    author='Jack Scantlebury',
    packages=[''],
    package_dir={'': ''},
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'wandb',
        'yaml',
        'pyyaml',
        'einops',
        'rdkit',
        'matplotlib',
        'openbabel',
        'scipy'
    ],
)
