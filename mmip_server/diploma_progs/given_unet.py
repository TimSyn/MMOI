from typing import List
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Input, concatenate, Softmax, Multiply
from tensorflow.keras.initializers import GlorotNormal

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def conv2d_block(
    inputs, 
    use_batch_norm=True, 
    filters=16, 
    kernel_size=(3,3), 
    activation='relu', 
    kernel_initializer=GlorotNormal(), 
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (inputs)
    if use_batch_norm:
        # c = tfa.layers.GroupNormalization(groups=16)(c)
        c = BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (c)
    if use_batch_norm:
        # c = tfa.layers.GroupNormalization(groups=16)(c)
        c = BatchNormalization()(c)
    return c

def conv2d_res_block(inputs, BN, filters, kernel_size=(3,3), activation='relu', padding='same'):    
    shortcut = Conv2D(filters, (1, 1), padding=padding) (inputs)
    shortcut = BatchNormalization()(shortcut)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=GlorotNormal(), padding=padding) (inputs)
    if BN:
        c = BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=GlorotNormal(), padding=padding) (c)
    if BN:
        c = BatchNormalization()(c)
    return c + shortcut

def res_unet(input_shape_1, input_shape_2, n_classes, BN=True, filters=16, n_layers=4):
    
    inputs = Input(input_shape_1)
    weight_inputs = Input(input_shape_2)
    x = inputs   
    down_layers = []
    for _ in range(n_layers):
        x = conv2d_res_block(inputs=x, BN=BN, filters=filters)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        filters = filters * 2 # double the number of filters with each layer
    x = conv2d_res_block(inputs=x, BN=BN, filters=filters)
    for conv_layer in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer 
        x = upsample_conv(filters, (2, 2), strides=(2, 2), padding='same') (x)
        x = concatenate([x, conv_layer])
        x = conv2d_res_block(inputs=x, BN=BN, filters=filters)
    
    x = Conv2D(n_classes, (1, 1)) (x)
    x = Multiply() ([x, weight_inputs[:, :, :, tf.newaxis]])
    # x = Multiply() ([x, tf.ones((3, 384, 384, 1))])
    outputs = Softmax() (x)
    
    model = Model(inputs=[inputs, weight_inputs], outputs=[outputs])

    return model