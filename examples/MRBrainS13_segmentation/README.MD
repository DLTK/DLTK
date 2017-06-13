# Brain tissue segmentation on the [MRBrainS13](http://mrbrains13.isi.uu.nl/) database

## Getting started
1. [Register](http://mrbrains13.isi.uu.nl/register.php) and download the challenge data to $DLTK_ROOT/data/MRBrainS13DataNii

2. Start a notebook server with
```shell
jupyter notebook --ip=* --port $MY_PORT
```

3. Run the tutorial.ipynb for a quick introduction

4. Run a full training and evaluation with 
```python
   python train.py [ARGS]
```

5. Launch a tensorboard server 
```shell
python -m tensorflow.tensorboard --logdir=/tmp/cifar10 --port=$MY_PORT
```

6. Open a browser and navigate to http://localhost:$MY_PORT/

## Additional notes
The notebook preprocess.ipynb attempts to reproduce the preprocessing in [2], is however work in progress. All results were generated from the original data.

## References
[1] A Mendrik et al. (2015); MRBrainS Challenge: Online evaluation framework for brain image segmentation in 3T MRI scans. Computational intelligence and neuroscience. 

[2] Stollenga et al. (2015); Parallel Multi-Dimensional LSTM, With Application to Fast Biomedical Volumetric Image Segmentation. NIPS 2015.

