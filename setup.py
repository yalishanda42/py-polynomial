#from distutils.core import setup
from setuptools import setup

setup(
  name='py-polynomial',
  packages=['polynomial'],
  version='0.2',
  license='MIT',
  description='Package handling mathematical single-variable polynomials.',
  author='Alexander Ignatov',
  author_email='yalishanda@abv.bg',
  url='https://github.com/allexks/py-polynomial',
  download_url='https://github.com/allexks/py-polynomial/archive/0.2.tar.gz',
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
