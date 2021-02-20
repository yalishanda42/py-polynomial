"""Setup script."""

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

REPO_URL = 'https://github.com/allexks/py-polynomial'
VERSION = '0.6.0'

setup(
  name='py-polynomial',
  packages=find_packages(exclude=("tests",)),
  version=VERSION,
  license='MIT',
  description='Package defining mathematical single-variable polynomials.',
  long_description=README,
  long_description_content_type="text/markdown",
  author='Alexander Ignatov',
  author_email='yalishanda@abv.bg',
  url=REPO_URL,
  download_url=f'{REPO_URL}/archive/{VERSION}.tar.gz',
  keywords=[
    'algebra',
    'polynomial',
    'polynomials',
    'mathematics',
    'maths',
    'derivative',
    'derivatives',
    'factor',
    'factors',
    'root',
    'roots',
    'terms',
    'coefficients',
    'quadratic',
    'linear',
    'sympy',
    'numpy',
  ],
  install_requires=[],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
