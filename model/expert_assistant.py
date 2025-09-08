from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Conv2D, Add, ZeroPadding2D, Dense, Flatten, AvgPool2D
from tensorflow.keras.layers import Input, Activation, Lambda, GRU, ZeroPadding1D, Bidirectional, Concatenate

#from tensorflow.keras.initializers import ones, zeros

import tensorflow as tf
import numpy as np



def temporal_shuffle(inputs):
    x = tf.reshape(inputs, shape = [-1, 1, 64, 2, 12])
    x = tf.transpose(x, perm = (0, 1, 2, 4, 3))
    out = tf.reshape(x, shape = [-1, 1, 64, 24])
    
    return out


def Expert_Assistant(input_length = 128, num_classes = 11):

    inputs = Input(shape = (2, input_length), name = 'input')

    # Expert
    x = Reshape((2, input_length, 1), name = 'reshape1')(inputs)
    x = ZeroPadding2D(((0, 0), (3, 4)), name = 'pad-b1')(x)
    x = Conv2D(75, (2, 8), activation = 'relu', name = 'conv-b1')(x)
    x = ZeroPadding2D((0, 2), name = 'pad-b2')(x)
    x = Conv2D(24, (1, 5), activation = 'relu', name = 'conv-b2')(x)
    x = AvgPool2D((1, 2), name = 'pool')(x)

    # Assistants
    conv1out = ZeroPadding2D(((0, 0), (3, 4)))(inputs[:, :, :(input_length//2), np.newaxis])
    conv1out = Conv2D(12, (2, 8), activation = 'relu', name = 'conv1-a1out')(conv1out)
    conv2out = ZeroPadding2D(((0, 0), (3, 4)))(inputs[:, :, (input_length//2):, np.newaxis])
    conv2out = Conv2D(12, (2, 8), activation = 'relu', name = 'conv1-a2out')(conv2out)
    convout = Concatenate(axis = -1)([conv1out, conv2out])

    convout = Lambda(temporal_shuffle, name = 'shuff')(convout)

    conv1in = ZeroPadding2D((0, 2))(convout[:, :, :, :12])
    conv2in = ZeroPadding2D((0, 2))(convout[:, :, :, 12:])
    conv1out = Conv2D(4, (1, 5), activation = 'relu', name = 'conv2-a1out')(conv1in)
    conv2out = Conv2D(4, (1, 5), activation = 'relu', name = 'conv2-a2out')(conv2in)
    convout = Concatenate(axis = -1)([conv1out, conv2out])

    # RNN Components
    x = Concatenate(axis = -1)([x, convout])

    x = Reshape((input_length//2, 32), name = 'reshape2')(x) 
    x = GRU(64, name = 'gru')(x)
    
    outputs = Dense(num_classes, activation = 'softmax', name = 'dense_class')(x)

    return Model(inputs = inputs, outputs = outputs)