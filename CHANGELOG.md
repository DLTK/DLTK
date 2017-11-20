# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]



## [0.2]

### Added 
- download scripts for example data
- model zoo repo
- simple super-resolution network
- full example training and deploy scripts on medical images for regression, classification, segmentation, representation learning, super-resolution w/ downloadable pre-trained models.
- continuous testing, code linter and automatic doc generation

### Changed
- unet implementation now has convolutional blocks symmetrically in the encoder and decoder
- core modularisation based on AbstractModule replaced by tf.layer and tf.estimator
- core io AbstractReader class now wraps tf.contrib.data.Dataset, replacing v0.1 reader and queuing classes
- tutorials are now changed to explain how dltk modules can be used with tf 
- improved readme in terms of contributing, installation instructions
- switched to google docstrings and loose coding style

### Removed
- sonnet like modularisation: AbstractModule classes
- summary modules, instead relying on problem-specific tf.summaries in estimator model_fns
- examples based on natural images (MNIST, CIFAR10, etc.)
