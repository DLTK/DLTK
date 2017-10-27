# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]



## [0.2]

### Added 

### Changed
- unet implementation now has convolutional blocks symmetrically in the encoder and decoder
- core modularisation based on AbstractModule replaced by new tf.layer and tf.estimator classes
- core io AbstractReader class now wraps tf.contrib.data.Dataset, replacing v0.1 reader and queuing classes

### Removed
- sonnet like modularisation: AbstractModule classes
- summary modules, instead relying on custom summaries in estimator model_fns

