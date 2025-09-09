# An Expert-Assistant Network for Efficient Automatic Modulation Recognition

This code is the Tensorflow implementation of the Expert-Assistant (E-A) network for efficient automatic modulation recognition (AMR).

## Requirements
- Python 3.x
- matplotlib
- Tensorflow 2.13


## Getting Started

### Dataset and Configurations
- Download the RadioML2016.10a dataset from https://www.deepsig.ai/datasets/ and put the dataset into the "dataset" directory.
- Set configurations in `run_tf.py` (training and testing) or `test_tf.py` (testing only).

### Training
To train The E-A network on RML16, run:
```cmd
python train_tf.py
```
After training, the best network will be automatically tested (in the same way as `test_tf.py`).

### Testing
To test the trained E-A network on RML16, run:
```cmd
python test_tf.py
```

We have provided an example trained E-A network in the "logs" directory.
