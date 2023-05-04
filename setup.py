from setuptools import setup, find_packages

setup(
    name='polysplit',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'shapely',
        'numpy',
        'scikit-learn-extra',
        'matplotlib',
        'networkx',
    ],
    extras_require={
        'tests': [
            'unittest',
        ],
    },
)
