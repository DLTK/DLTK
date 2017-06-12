# Brain tissue segmentation on the [MRBrainS13](http://mrbrains13.isi.uu.nl/) database

## Getting started
* [Register](http://mrbrains13.isi.uu.nl/register.php) and download the challenge data to $DLTK_ROOT/data/MRBrainS13DataNii
* Launch an ipython notebook with $DLTK_ROOT/start_ipynb_serv.sh
* Run the tutorial.ipynb for a quick introduction

* Run a full training and evaluation with 
```python
   python train.py [ARGS]
```
You can launch a tensorboard webserver (see $DLTK_ROOT/launch_tensorboard.sh) to track the training progress and performance

## Additional notes
The notebook preprocess.ipynb attempts to reproduce the preprocessing in [2], is however work in progress. All results were generated from the original data.

## References
[1] A Mendrik et al. (2015); MRBrainS Challenge: Online evaluation framework for brain image segmentation in 3T MRI scans. Computational intelligence and neuroscience. 

[2] Stollenga et al. (2015); Parallel Multi-Dimensional LSTM, With Application to Fast Biomedical Volumetric Image Segmentation. NIPS 2015.


## TODO:
* Add new reader to the tutorial nb, clean, comment and test
