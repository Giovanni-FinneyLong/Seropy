# from distutils.core import setup
from setuptools import setup, find_packages

# install = [
#     'numpy',
#     'matplotlib',
#     'scipy',
#     'scikit-learn',
#     'vispy',
#     'munkres'
# ]


setup(
    name='Serotonin',
    version='1.0',
    install_requires=['numpy', 'vispy', 'munkres', 'scipy', 'pandas', 'matplotlib'],
    packages=find_packages(),
    url='https://bitbucket.org/gfinneylong/serotonin',
    license='',
    author='Giovanni Finney-Long',
    author_email='gfinneylong@gmail.com',
    description='Python Blob Analysis'
)
