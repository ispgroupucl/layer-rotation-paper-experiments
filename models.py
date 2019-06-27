'''
Base functions to create VGG-, resnet- and WideResnet-style networks
'''

import os

import numpy as np

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, add, BatchNormalization
from keras.layers import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, LeakyReLU
from keras.regularizers import l2

#==============================================================================
# VGG-style CNN
#==============================================================================

def VGG(input_shape, nbstages, nblayers, nbfilters,nbclasses,weight_decay=0., 
        kernel_constraint = None, kernel_initializer='glorot_uniform', include_top = True, use_batchnorm = True,
        batchnorm_training = True, use_bias = True, act = 'relu', dropout = 0., kernel_size = (3,3),
        batchnorm_momentum = 0.99, use_skips = False):
    '''
    VGG-style convolutional neural network
    
    nbstages is the number of spatial dimension levels
    nblayers is a list with nbstages elements containing the 
        number of convolutional layers per stage
    nbfilters is a list of size sum(nbstages) with the 
        number of filters per convolutional layer in a stage
    
    kernel_constraint only applied on convolutional layers
    
    uses batchnorm after each Convolutional layer (non-linearity included)
    '''     
    if K.image_data_format() == 'channels_last':
        if len(input_shape) == 2:
            input_shape = input_shape + (3,)
        channel_axis = -1
    elif K.image_data_format() == 'channels_first':
        if len(input_shape) == 2:
            input_shape = (3,) + input_shape
        channel_axis = 1
    
    if len(nblayers) != nbstages:
        raise ValueError('nblayers should contain one element per stage.')
    if len(nbfilters) != nbstages:
        raise ValueError('nbfilters should contain one element per stage.')
    
    regularizer = None
    if weight_decay > 0.:
        regularizer = l2(weight_decay)
    
    input_model = Input(shape = input_shape)
    x = input_model
    
    layer_counter = 0
    for s in range(nbstages):
        for l in range(nblayers[s]):
            inp = x
            
            x = Conv2D(nbfilters[s], kernel_size = kernel_size, padding = 'same',
                       name = 'stage'+str(s)+'_layer'+str(l)+'_conv',
                       kernel_constraint = kernel_constraint,
                       kernel_initializer = kernel_initializer,
                       kernel_regularizer=regularizer,
                       use_bias = use_bias)(x)
            
            if dropout > 0.:
                x = Dropout(dropout)(x)
                
            if act is not 'leaky':
                x = Activation('relu', name = 'stage'+str(s)+'_layer'+str(l)+'_relu')(x)
            else:
                x = LeakyReLU(alpha = 0.3, name = 'stage'+str(s)+'_layer'+str(l)+'_relu')(x)
            
            if use_batchnorm:
                x = BatchNormalization(axis = channel_axis, name = 'stage'+str(s)+'_layer'+str(l)+'_batch',
                                       center = batchnorm_training, scale = batchnorm_training,
                                       momentum = batchnorm_momentum)(x)
            
            if use_skips and l!=0: # don't consider conv layers where the number of feature maps changes
                x = add([x,inp])
            
            layer_counter += 1
        
        if s != nbstages-1: # do not perform maxpooling in last stage
            x = MaxPooling2D((2,2),strides = (2,2), name = 'stage'+str(s)+'_pool')(x)
    if include_top:
        x = GlobalAveragePooling2D(name = 'global_pool')(x)
        x = Dense(nbclasses,name = 'last_dense', kernel_initializer = kernel_initializer, use_bias = use_bias,
                  kernel_regularizer=regularizer)(x)
        x = Activation('softmax',name = 'predictions')(x)
    
    return Model(input_model,x)

#==============================================================================
# ResNet-style, code taken from keras examples: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
#==============================================================================

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 use_bias = True,
                 batchnorm_training = True,
                 name = '',
                 weight_decay=0.): #1e-4
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    regularizer = l2(weight_decay) if weight_decay >0. else None
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  use_bias = use_bias,
                  name = name+'_conv',
                  kernel_regularizer=regularizer)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(center = batchnorm_training, scale = batchnorm_training, name = name + '_batch')(x)
        if activation is not None:
            x = Activation(activation, name = name+'_act')(x)
    else:
        if batch_normalization:
            x = BatchNormalization(center = batchnorm_training, scale = batchnorm_training, name = name + '_batch')(x)
        if activation is not None:
            x = Activation(activation, name = name+'_act')(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, use_bias = True, batchnorm_training = True, weight_decay = 0.):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, use_bias = use_bias, batchnorm_training = batchnorm_training, name = 'first',weight_decay = weight_decay)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             use_bias = use_bias, batchnorm_training = batchnorm_training,
                             name = str(stack)+'_'+str(res_block)+'_1',
                             weight_decay = weight_decay)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             use_bias = use_bias, batchnorm_training = batchnorm_training,
                             name = str(stack)+'_'+str(res_block)+'_2',
                             weight_decay = weight_decay)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 use_bias = use_bias, batchnorm_training = batchnorm_training,
                                 name = str(stack)+'_'+str(res_block)+'_strided',
                                 weight_decay = weight_decay)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal',
                    use_bias = use_bias,
                    name = 'last_dense')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

#==============================================================================
# VGG model from pytorchblog: http://torch.ch/blog/2015/07/30/cifar.html
# Was also used by "The marginal value of adaptive gradient methods in machine learning"
#==============================================================================

def VGG_pytorchBlogStyle(input_shape, nbstages, nblayers, nbfilters,nbclasses,weight_decay=0., 
        kernel_constraint = None, kernel_initializer='glorot_uniform', include_top = True, use_batchnorm = True,
        batchnorm_training = True, use_bias = True, act = 'relu', dropout = True, kernel_size = (3,3),
        batchnorm_position = 'before'):
    '''
    nbstages is the number of spatial dimension levels
    nblayers is a list with nbstages elements containing the 
        number of convolutional layers per stage
    nbfilters is a list of size sum(nbstages) with the 
        number of filters per convolutional layer in a stage
    
    kernel_constraint only applied on convolutional layers
    
    uses batchnorm after or before non-linearity
    '''    
    if K.image_data_format() == 'channels_last':
        if len(input_shape) == 2:
            input_shape = input_shape + (3,)
        channel_axis = -1
    elif K.image_data_format() == 'channels_first':
        if len(input_shape) == 2:
            input_shape = (3,) + input_shape
        channel_axis = 1
    
    if len(nblayers) != nbstages:
        raise ValueError('nblayers should contain one element per stage.')
    if len(nbfilters) != nbstages:
        raise ValueError('nbfilters should contain one element per stage.')
        
    if batchnorm_position not in ['after','before']:
        raise ValueError('batchnorm_position argument should be either \'after\' or \'before\'')
    
    regularizer = None
    if weight_decay > 0.:
        regularizer = l2(weight_decay)
    
    input_model = Input(shape = input_shape)
    x = input_model
    
    layer_counter = 0
    for s in range(nbstages):
        for l in range(nblayers[s]):
            x = Conv2D(nbfilters[s], kernel_size = kernel_size, padding = 'same',
                       name = 'stage'+str(s)+'_layer'+str(l)+'_conv',
                       kernel_constraint = kernel_constraint,
                       kernel_initializer = kernel_initializer,
                       kernel_regularizer=regularizer,
                       use_bias = use_bias)(x)
            
            if use_batchnorm and batchnorm_position == 'before':
                x = BatchNormalization(axis = channel_axis, name = 'stage'+str(s)+'_layer'+str(l)+'_batch',
                                       center = batchnorm_training, scale = batchnorm_training)(x)
                
            if act is not 'leaky':
                x = Activation('relu', name = 'stage'+str(s)+'_layer'+str(l)+'_relu')(x)
            else:
                x = LeakyReLU(alpha = 0.3, name = 'stage'+str(s)+'_layer'+str(l)+'_relu')(x)
                
            if use_batchnorm and batchnorm_position == 'after':
                x = BatchNormalization(axis = channel_axis, name = 'stage'+str(s)+'_layer'+str(l)+'_batch',
                                       center = batchnorm_training, scale = batchnorm_training)(x)
            
            if l<nblayers[s]-1 and dropout:
                if s == 0:
                    x = Dropout(0.3)(x)
                else:
                    x = Dropout(0.4)(x)
                        
            layer_counter += 1
        
        x = MaxPooling2D((2,2),strides = (2,2), name = 'stage'+str(s)+'_pool')(x)
    
    if include_top:
        x = Flatten()(x)
        if dropout:
            x = Dropout(0.5)(x)
        x = Dense(512, kernel_initializer = kernel_initializer, use_bias = use_bias,kernel_regularizer=regularizer, name = 'dense1')(x)
        if use_batchnorm and batchnorm_position == 'before':
            x = BatchNormalization(axis = channel_axis,center = batchnorm_training, scale = batchnorm_training)(x)
        x = Activation('relu')(x)
        if use_batchnorm and batchnorm_position == 'after':
            x = BatchNormalization(axis = channel_axis,center = batchnorm_training, scale = batchnorm_training)(x)
        if dropout:
            x = Dropout(0.5)(x)
        x = Dense(nbclasses,name = 'last_dense', kernel_initializer = kernel_initializer, use_bias = use_bias,
                 kernel_regularizer=regularizer)(x)
        x = Activation('softmax',name = 'predictions')(x)
    
    return Model(input_model,x)
    

#==============================================================================
# WideResnet model: https://arxiv.org/abs/1605.07146
#==============================================================================

def block(inp,nbfilters,dropout,weight_decay,channel_axis,subsample = (1,1), batchnorm_training = True, use_bias = True): 
    x = inp
    
    for i in [1,2]:
        x = BatchNormalization(axis = channel_axis,center = batchnorm_training, scale = batchnorm_training)(x)
        x = Activation('relu')(x)
        
        if dropout>0. and i==2:
            x = Dropout(dropout)(x)
        
        x = ZeroPadding2D((1,1))(x)
        if subsample is not None and i==1:
            x = Conv2D(nbfilters,(3,3),strides=subsample,kernel_regularizer=l2(weight_decay), use_bias = use_bias)(x)
        else:
            x = Conv2D(nbfilters,(3,3),kernel_regularizer=l2(weight_decay), use_bias = use_bias)(x)
    
    if subsample==(1,1) and inp._keras_shape[channel_axis] == nbfilters: # checks for subsampling or change in nb of filters
        return add([x,inp])
    else:
        return add([x,Conv2D(nbfilters,(1,1),strides = subsample,kernel_regularizer=l2(weight_decay), use_bias = use_bias)(inp)])

def stage(x, nbfilters, N, dropout, weight_decay, channel_axis, subsample = True, batchnorm_training = True, use_bias = True):
    if subsample:
        x = block(x,nbfilters, dropout, weight_decay, channel_axis, 
                  subsample = (2,2), batchnorm_training = batchnorm_training, use_bias = use_bias)
    else: 
        x = block(x,nbfilters, dropout, weight_decay, channel_axis, 
                  batchnorm_training = batchnorm_training, use_bias = use_bias)
    for i in range(1,N):
        x = block(x,nbfilters, dropout, weight_decay, channel_axis, 
                  batchnorm_training = batchnorm_training, use_bias = use_bias)
    return x

# Wide ResNet architecture
# Contains 3 stages
# arguments are lists with one parameter per stage
# input conv nbfilter is always 16
def WideResNet(nbfilters,nbblocks,dropout,weight_decay,nb_classes, batchnorm_training = True, use_bias = True):
    if K.image_data_format() == 'channels_last':
        input_model = Input(shape = (32,32,3))
        channel_axis = -1
    elif K.image_data_format() == 'channels_first':
        input_model = Input(shape = (3,32,32))
        channel_axis = 1
    
    # input convolution
    x = ZeroPadding2D((1,1))(input_model)
    x = Conv2D(16, (3, 3),kernel_regularizer=l2(weight_decay), use_bias = use_bias)(x)
    
    # stage 1, 32x32
    x = stage(x,nbfilters[0],nbblocks[0], dropout, weight_decay, channel_axis, subsample = False,batchnorm_training = batchnorm_training, use_bias = use_bias)
    
    # stage 2, 16x16
    x = stage(x,nbfilters[1],nbblocks[1], dropout, weight_decay, channel_axis,batchnorm_training = batchnorm_training, use_bias = use_bias)
    
    # stage 3, 8x8
    x = stage(x,nbfilters[2],nbblocks[2], dropout, weight_decay, channel_axis,batchnorm_training = batchnorm_training, use_bias = use_bias)
    
    x = BatchNormalization(axis = channel_axis,center = batchnorm_training, scale = batchnorm_training)(x)
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay), use_bias = use_bias)(x)
    x = Activation('softmax')(x)
    
    return Model(input_model,x)