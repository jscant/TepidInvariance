from setuptools import setup

setup(
    name='tepid_invariance',
    version='0.0.2',
    description='SE(3)-equivariant neural networks for hotspot prediction.',
    author='Jack Scantlebury',
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'wandb',
        'pyyaml',
        'einops',
        'matplotlib',
        'pybel',
        'scipy'
    ],
)
