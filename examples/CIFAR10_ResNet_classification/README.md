# ResNet classification on the CIFAR-10 dataset


### Run the example

1. Download the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:

```shell
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz 
tar -xzvf cifar-10-binary.tar.gz

```

2. Extract it to $DLTK_ROOT/data

3. Run the example
```shell
python train.py [ARGS]
```

4. Launch a tensorboard server 
```shell
python -m tensorflow.tensorboard --logdir=/tmp/cifar10 --port=$MY_PORT
```

5. Open a browser and navigate to http://localhost:$MY_PORT/


### Acknowledgements
* [CIFAR10 data](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)
* [Tensorflow models](https://github.com/tensorflow/models/tree/master/resnet)