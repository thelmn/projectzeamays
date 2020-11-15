# %%
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, Flatten, Dense
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf

# %%

def unet1():
    """ UNet model 1.
    Architecture:
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 256, 256, 3) 0                                            
    conv2d (Conv2D)                 (None, 256, 256, 32) 896         input_1[0][0]                    
    conv2d_1 (Conv2D)               (None, 256, 256, 32) 9248        conv2d[0][0]                     
    max_pooling2d (MaxPooling2D)    (None, 128, 128, 32) 0           conv2d_1[0][0]                   
    conv2d_2 (Conv2D)               (None, 128, 128, 64) 18496       max_pooling2d[0][0]              
    conv2d_3 (Conv2D)               (None, 128, 128, 64) 36928       conv2d_2[0][0]                   
    max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 64)   0           conv2d_3[0][0]                   
    conv2d_4 (Conv2D)               (None, 64, 64, 128)  73856       max_pooling2d_1[0][0]            
    conv2d_5 (Conv2D)               (None, 64, 64, 128)  147584      conv2d_4[0][0]                   
    max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 128)  0           conv2d_5[0][0]                   
    conv2d_6 (Conv2D)               (None, 32, 32, 256)  295168      max_pooling2d_2[0][0]            
    conv2d_7 (Conv2D)               (None, 32, 32, 256)  590080      conv2d_6[0][0]                   
    max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 256)  0           conv2d_7[0][0]                   
    conv2d_8 (Conv2D)               (None, 16, 16, 512)  1180160     max_pooling2d_3[0][0]            
    conv2d_9 (Conv2D)               (None, 16, 16, 512)  2359808     conv2d_8[0][0]                   
    conv2d_transpose (Conv2DTranspo (None, 32, 32, 256)  1179904     conv2d_9[0][0]                   
    concatenate (Concatenate)       (None, 32, 32, 512)  0           conv2d_7[0][0]                   
                                                                     conv2d_transpose[0][0]           
    conv2d_10 (Conv2D)              (None, 32, 32, 256)  1179904     concatenate[0][0]                
    conv2d_11 (Conv2D)              (None, 32, 32, 256)  590080      conv2d_10[0][0]                  
    conv2d_transpose_1 (Conv2DTrans (None, 64, 64, 128)  295040      conv2d_11[0][0]                  
    concatenate_1 (Concatenate)     (None, 64, 64, 256)  0           conv2d_5[0][0]                   
                                                                     conv2d_transpose_1[0][0]         
    conv2d_12 (Conv2D)              (None, 64, 64, 128)  295040      concatenate_1[0][0]              
    conv2d_13 (Conv2D)              (None, 64, 64, 64)   73792       conv2d_12[0][0]                  
    conv2d_14 (Conv2D)              (None, 64, 64, 3)    195         conv2d_13[0][0]                  
    ==================================================================================================
    Total params: 8,326,179
    Trainable params: 8,326,179
    Non-trainable params: 0
    __________________________________________________________________________________________________
"""
    # UNet
    i = Input(shape=(256,256,3))

    # lvl1
    l = Conv2D(32, (3, 3), padding='same', activation='relu')(i)
    l1 = Conv2D(32, (3, 3), padding='same', activation='relu')(l)
    l = MaxPool2D((2, 2))(l1)

    # lvl2
    l = Conv2D(64, (3, 3), padding='same', activation='relu')(l)
    l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l)
    l = MaxPool2D((2, 2))(l2)

    # lvl3
    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l3 = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l = MaxPool2D((2, 2))(l3)

    # lvl4
    l = Conv2D(256, (3, 3), padding='same', activation='relu')(l)
    l4 = Conv2D(256, (3, 3), padding='same', activation='relu')(l)
    l = MaxPool2D((2, 2))(l4)

    # lvl5
    l = Conv2D(512, (3, 3), padding='same', activation='relu')(l)
    l5 = Conv2D(512, (3, 3), padding='same', activation='relu')(l)

    l = Conv2DTranspose(256, (3, 3), strides=(2,2), padding='same', activation='relu')(l5)

    l = Concatenate(axis=-1)([l4, l])
    l = Conv2D(256, (3, 3), padding='same', activation='relu')(l)
    _l4 = Conv2D(256, (3, 3), padding='same', activation='relu')(l)

    l = Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same', activation='relu')(_l4)

    l = Concatenate(axis=-1)([l3, l])
    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(64, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(3, (1, 1), padding='same')(l)

    model = Model(inputs=i, outputs=l)
    return model

# %%
def unet2():
    """ UNet model 2. Reduced filters (max 128) vs unet1
    Architecture:
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 256, 256, 3) 0                                            
    conv2d (Conv2D)                 (None, 256, 256, 32) 896         input_1[0][0]                    
    conv2d_1 (Conv2D)               (None, 256, 256, 32) 9248        conv2d[0][0]                     
    max_pooling2d (MaxPooling2D)    (None, 128, 128, 32) 0           conv2d_1[0][0]                   
    conv2d_2 (Conv2D)               (None, 128, 128, 64) 18496       max_pooling2d[0][0]              
    conv2d_3 (Conv2D)               (None, 128, 128, 64) 36928       conv2d_2[0][0]                   
    max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 64)   0           conv2d_3[0][0]                   
    conv2d_4 (Conv2D)               (None, 64, 64, 128)  73856       max_pooling2d_1[0][0]            
    conv2d_5 (Conv2D)               (None, 64, 64, 128)  147584      conv2d_4[0][0]                   
    max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 128)  0           conv2d_5[0][0]                   
    conv2d_6 (Conv2D)               (None, 32, 32, 128)  147584      max_pooling2d_2[0][0]            
    conv2d_7 (Conv2D)               (None, 32, 32, 128)  147584      conv2d_6[0][0]                   
    max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 128)  0           conv2d_7[0][0]                   
    conv2d_8 (Conv2D)               (None, 16, 16, 128)  147584      max_pooling2d_3[0][0]            
    conv2d_9 (Conv2D)               (None, 16, 16, 128)  147584      conv2d_8[0][0]                   
    conv2d_transpose (Conv2DTranspo (None, 32, 32, 128)  147584      conv2d_9[0][0]                   
    concatenate (Concatenate)       (None, 32, 32, 256)  0           conv2d_7[0][0]                   
                                                                     conv2d_transpose[0][0]           
    conv2d_10 (Conv2D)              (None, 32, 32, 128)  295040      concatenate[0][0]                
    conv2d_11 (Conv2D)              (None, 32, 32, 128)  147584      conv2d_10[0][0]                  
    conv2d_transpose_1 (Conv2DTrans (None, 64, 64, 128)  147584      conv2d_11[0][0]                  
    concatenate_1 (Concatenate)     (None, 64, 64, 256)  0           conv2d_5[0][0]                   
                                                                     conv2d_transpose_1[0][0]         
    conv2d_12 (Conv2D)              (None, 64, 64, 128)  295040      concatenate_1[0][0]              
    conv2d_13 (Conv2D)              (None, 64, 64, 64)   73792       conv2d_12[0][0]                  
    conv2d_14 (Conv2D)              (None, 64, 64, 3)    195         conv2d_13[0][0]                  
    ==================================================================================================
    Total params: 1,984,163
    Trainable params: 1,984,163
    Non-trainable params: 0
    __________________________________________________________________________________________________
"""
    # UNet
    i = Input(shape=(256,256,3))

    # lvl1
    l = Conv2D(32, (3, 3), padding='same', activation='relu')(i)
    l1 = Conv2D(32, (3, 3), padding='same', activation='relu')(l)
    l = MaxPool2D((2, 2))(l1)

    # lvl2
    l = Conv2D(64, (3, 3), padding='same', activation='relu')(l)
    l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l)
    l = MaxPool2D((2, 2))(l2)

    # lvl3
    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l3 = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l = MaxPool2D((2, 2))(l3)

    # lvl4
    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l4 = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l = MaxPool2D((2, 2))(l4)

    # lvl5
    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu')(l)

    l = Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same', activation='relu')(l5)

    l = Concatenate(axis=-1)([l4, l])
    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    _l4 = Conv2D(128, (3, 3), padding='same', activation='relu')(l)

    l = Conv2DTranspose(128, (3, 3), strides=(2,2), padding='same', activation='relu')(_l4)

    l = Concatenate(axis=-1)([l3, l])
    l = Conv2D(128, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(64, (3, 3), padding='same', activation='relu')(l)
    l = Conv2D(3, (1, 1), padding='same')(l)

    model = Model(inputs=i, outputs=l)
    return model

# %%
def conv2d_bn(input, n_filters, filter_size=(3,3), padding='same', activation='relu', bn=True, name=None):
    conv2d_name = None
    if activation is None and not bn:
        conv2d_name = name
    l = Conv2D(n_filters, filter_size, padding=padding, use_bias=False, name=conv2d_name)(input)

    bn_name = None
    if activation is None:
        bn_name = name
    if bn:
        l = layers.BatchNormalization(name=bn_name)(l)

    if activation is not None:
        l = layers.Activation('relu', name=name)(l)
        layers.Layer()
    return l

def conv2d_t_bn(input, n_filters, filter_size=(3,3), strides=(2,2), padding='same', activation='relu', bn=True):
    l = Conv2DTranspose(n_filters, filter_size, strides=strides, padding=padding, use_bias=False)(input)
    if bn:
        l = layers.BatchNormalization()(l)
    if activation is not None:
        l = layers.Activation('relu')(l)
    return l

# %%
def unet3_encoder(i):
    # lvl1
    l = conv2d_bn(i, 32, (3, 3))
    l1 = conv2d_bn(l, 32, (3, 3))
    l = MaxPool2D((2, 2))(l1)

    # lvl2
    l = conv2d_bn(l, 64, (3, 3))
    l2 = conv2d_bn(l, 64, (3, 3))
    l = MaxPool2D((2, 2))(l2)

    # lvl3
    l = conv2d_bn(l, 128, (3, 3))
    l3 = conv2d_bn(l, 128, (3, 3), name='conv2d_bn_enc3_64x64x128')
    l = MaxPool2D((2, 2))(l3)

    # lvl4
    l = conv2d_bn(l, 128, (3, 3))
    l4 = conv2d_bn(l, 128, (3, 3), name='conv2d_bn_enc4_32x32x128')
    l = MaxPool2D((2, 2))(l4)

    # lvl5
    l = conv2d_bn(l, 128, (3, 3))
    l5 = conv2d_bn(l, 128, (3, 3), name='conv2d_bn_enc5_16x16x128')

    return l3, l4, l5

def unet3_decoder(l3, l4, l5):
    l = conv2d_t_bn(l5, 128, (3, 3), strides=(2,2))

    l = Concatenate(axis=-1)([l4, l])

    l = conv2d_bn(l, 128, (3, 3))
    _l4 = conv2d_bn(l, 128, (3, 3))

    l = conv2d_t_bn(_l4, 128, (3, 3), strides=(2,2))

    l = Concatenate(axis=-1)([l3, l])

    l = conv2d_bn(l, 128, (3, 3))
    l = conv2d_bn(l, 64, (3, 3))
    l = conv2d_bn(l, 3, (1, 1), bn=False, activation=None)
    return l

# %%
def unet3():
    """ UNet model 3. Reduced filters (max 128) vs unet1. With batchNorm
    Architecture:
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 256, 256, 3) 0                                            
    conv2d (Conv2D)                 (None, 256, 256, 32) 896         input_1[0][0]                    
    conv2d_1 (Conv2D)               (None, 256, 256, 32) 9248        conv2d[0][0]                     
    max_pooling2d (MaxPooling2D)    (None, 128, 128, 32) 0           conv2d_1[0][0]                   
    conv2d_2 (Conv2D)               (None, 128, 128, 64) 18496       max_pooling2d[0][0]              
    conv2d_3 (Conv2D)               (None, 128, 128, 64) 36928       conv2d_2[0][0]                   
    max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 64)   0           conv2d_3[0][0]                   
    conv2d_4 (Conv2D)               (None, 64, 64, 128)  73856       max_pooling2d_1[0][0]            
    conv2d_5 (Conv2D)               (None, 64, 64, 128)  147584      conv2d_4[0][0]                   
    max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 128)  0           conv2d_5[0][0]                   
    conv2d_6 (Conv2D)               (None, 32, 32, 128)  147584      max_pooling2d_2[0][0]            
    conv2d_7 (Conv2D)               (None, 32, 32, 128)  147584      conv2d_6[0][0]                   
    max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 128)  0           conv2d_7[0][0]                   
    conv2d_8 (Conv2D)               (None, 16, 16, 128)  147584      max_pooling2d_3[0][0]            
    conv2d_9 (Conv2D)               (None, 16, 16, 128)  147584      conv2d_8[0][0]                   
    conv2d_transpose (Conv2DTranspo (None, 32, 32, 128)  147584      conv2d_9[0][0]                   
    concatenate (Concatenate)       (None, 32, 32, 256)  0           conv2d_7[0][0]                   
                                                                     conv2d_transpose[0][0]           
    conv2d_10 (Conv2D)              (None, 32, 32, 128)  295040      concatenate[0][0]                
    conv2d_11 (Conv2D)              (None, 32, 32, 128)  147584      conv2d_10[0][0]                  
    conv2d_transpose_1 (Conv2DTrans (None, 64, 64, 128)  147584      conv2d_11[0][0]                  
    concatenate_1 (Concatenate)     (None, 64, 64, 256)  0           conv2d_5[0][0]                   
                                                                     conv2d_transpose_1[0][0]         
    conv2d_12 (Conv2D)              (None, 64, 64, 128)  295040      concatenate_1[0][0]              
    conv2d_13 (Conv2D)              (None, 64, 64, 64)   73792       conv2d_12[0][0]                  
    conv2d_14 (Conv2D)              (None, 64, 64, 3)    195         conv2d_13[0][0]                  
    ==================================================================================================
    Total params: 1,984,163
    Trainable params: 1,984,163
    Non-trainable params: 0
    __________________________________________________________________________________________________
"""
    # UNet
    i = Input(shape=(256,256,3))

    l3, l4, l5 = unet3_encoder(i)
    l = unet3_decoder(l3, l4, l5)
 
    model = Model(inputs=i, outputs=l)
    return model

def unet3_enc_classifier():
    i = Input(shape=(256,256,3))

    _, _, l5 = unet3_encoder(i)

    l = MaxPool2D((2, 2))(l5)

    l = conv2d_bn(l, 64, (3,3), padding='valid')
    l = Flatten()(l)
    l = Dense(1, activation='sigmoid')(l)

    model = Model(inputs=i, outputs=l)
    return model



