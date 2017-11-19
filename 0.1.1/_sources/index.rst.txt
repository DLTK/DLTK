:github_url: https://github.com/dltk/dltk

.. toctree::
   :hidden:

   Home <self>
   user_guide/getting_started
   user_guide/module
   user_guide/model
   user_guide/reader
   user_guide/usage

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Package Reference

   api/dltk

DLTK documentation
==================

DLTK is a neural networks toolkit written in python, on top of `Tensorflow <https://github.com/tensorflow/tensorflow>`_. Its modular architecture is closely inspired by `sonnet <https://github.com/deepmind/sonnet>`_ and it was developed to enable fast prototyping and ensure reproducibility in image analysis applications, with a particular focus on medical imaging. Its goal is to provide the community with state of the art methods and models and to accelerate research in this exciting field.

Road map
--------

Over the course of the next months we will add more content to DLTK. This road map outlines the immediate plans for what you will be seeing in DLTK soon:

Medical model zoo
  Pre-trained models on medical images and deploy scripts
  
Core
  Losses: Dice loss, frequency reweighted losses, adversial training
  Normalisation: layer norm, weight norm

Network architectures
  deepmedic, densenet, VGG, super-resolution networks

Other
  Augmentation via elastic deformations
  Sampling with fixed class frequencies
  


Installation
------------

DLTK uses the following dependencies:

* numpy
* scipy
* Tensorflow: `Installation Instructions <https://www.tensorflow.org/install/>`_

Use *pip* to install **DLTK**::

  pip install dltk
  
Contact
-------

Twitter: `@dltk_ <https://twitter.com/dltk_>`_  

Source on `github.com <https://github.com/DLTK/DLTK>`_  

Core Team
---------

`Martin Rajchl <http://www.imperial.ac.uk/people/m.rajchl>`_ [`github <https://github.com/mrajchl>`_, `twitter <https://twitter.com/m_rajchl>`_]

`Nick Pawlowski <http://nickpawlowski.de/>`_ [`github <https://github.com/pawni>`_, `twitter <https://twitter.com/pwnic>`_]

`Ira Ktena <https://biomedia.doc.ic.ac.uk/person/ira-ktena/>`_ [`github <https://github.com/sk1712>`_, `twitter <https://twitter.com/s0f1ra>`_]  

`Matt Lee <https://biomedia.doc.ic.ac.uk/person/matthew-lee/>`_ [`github <https://github.com/mauinz>`_]

`BioMedIA group <https://biomedia.doc.ic.ac.uk/>`_, Dept. of Computing, Imperial College London

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
