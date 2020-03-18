"""Setup script."""

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
  name='py-polynomial',
  packages=find_packages(exclude=("tests",)),
  version='0.4.1',
  license='MIT',
  description='Package defining mathematical single-variable polynomials.',
  long_description=README,
  long_description_content_type="text/markdown",
  author='Alexander Ignatov',
  author_email='yalishanda@abv.bg',
  url='https://github.com/allexks/py-polynomial',
  download_url='https://github.com/allexks/py-polynomial/archive/0.4.1.tar.gz',
  keywords=['polynomial', 'maths', 'derivative', 'roots', 'algebra', 'linear'],
  install_requires=[],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
