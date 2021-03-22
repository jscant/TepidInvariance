from distutils.core import setup

setup(
    packages=['tepid_invariance'],
    name='tepid_invariance',
    version='0.0.3',
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
        'scipy'
        'openbabel',
        'plip'
    ],
)
