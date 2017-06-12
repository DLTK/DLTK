#!/usr/bin/env python

from distutils.core import setup

setup(name='dltk',
      version='0.1',
      description='Deep Learning Toolkit for Medical Image Analysis',
      author='Martin Rajchl, Nick Pawlowski',
      #author_email='gward@python.net',
      #url='https://www.python.org/sigs/distutils-sig/',
      packages=['dltk'],
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'sklearn', 'tensorflow-gpu', 'SimpleITK', 'jupyter']
     )
