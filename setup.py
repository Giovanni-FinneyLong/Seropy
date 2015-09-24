# from distutils.core import setup
from setuptools import setup, find_packages

install = [
    'numpy',
    'matplotlib',
    'scipy',
    'scikit-learn',
    'vispy',
    'munkres'
]


setup(
    name='Serotonin',
    version='.1',
    install_requires=install,
    packages=[''],
    package_dir={'': 'image-analysis'},
    url='https://bitbucket.org/gfinneylong/serotonin',
    license='',
    author='gio',
    author_email='gfinneylong@gmail.com',
    description='Python Blob Analysis'
)
