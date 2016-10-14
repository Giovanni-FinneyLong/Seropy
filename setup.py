# from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='Seropy',
    version='1.1.1',
    install_requires=['numpy', 'vispy', 'munkres', 'scipy', 'pandas', 'matplotlib', 'Goulib'],
    packages=find_packages(),
    url='https://bitbucket.org/gfinneylong/serotonin',
    license='',
    author='Giovanni Finney-Long',
    author_email='gfinneylong@gmail.com',
    description='Package for analysis and 3d-reconstruction of objects from laser confocal scans'
)
