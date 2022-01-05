#                    __
#                  / *_)
#       _.----. _ /../
#     /............./
# __/..(...|.(...|
# /__.-|_|--|_|
#
# Christos Kyrkou, PhD
# 2019

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Add, Input, Concatenate, Layer,Permute
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, AveragePooling2D, \
    SeparableConv2D, GlobalAveragePooling2D, Multiply
from tensorflow.keras.layers import Conv2DTranspose, Cropping2D,DepthwiseConv2D
from tensorflow.keras.layers import Lambda, SpatialDropout2D, LeakyReLU, UpSampling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import load_model, save_model


from tensorflow.keras.applications.resnet50 import ResNet50

import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers

from tensorflow.keras.activations import softmax

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


# from models.net_models import *
# from models.MiniGoogLeNet import *
# from models.sqeezenet import *
# from models.YOLOv2 import *

import sys

sys.path.append("..")
from gen_utils import change_model


ki = initializers.he_normal()

kid = initializers.RandomNormal(mean=0.0, stddev=0.02)

kr = regularizers.l2(5e-4)




def custom_preprocess(x):
    x = x/255.

    return x


def get_preprocess_method(net_name):
    preprocess_input = None

    if (net_name == 'resnet' or net_name == 'resnet25'):
        from tensorflow.keras.applications.resnet50 import preprocess_input

    if (net_name == 'vgg'):
        from tensorflow.keras.applications.vgg16 import preprocess_input

    if (net_name == 'efficient'):
        from tensorflow.keras.applications.EfficientNetB0 import preprocess_input
    if (net_name == 'nasmobile'):
        from tensorflow.keras.applications.nasnet import preprocess_input
    if (net_name == 'mobileV2'):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    if (net_name == 'mobile'):
        from tensorflow.keras.applications.mobilenet import preprocess_input

    return preprocess_input


def add_SSD_Head(params, x, ks=1, stride=1):
    ki = initializers.RandomNormal(mean=0.0, stddev=0.01)


    kr = regularizers.l2(5e-4)

    x = Conv2D(params.FILTER_SSD, (ks, ks), kernel_initializer=ki,
               activation='linear', use_bias=False, strides=(stride, stride), name="last_layer")(x)

    x = BatchNormalization()(x)

    x = Reshape((params.GRID_H, params.GRID_W, params.BOX, params.CLASS + 4 + 1))(x)

    return x


def add_YOLO_Head(params, x, ks=1, stride=1, weight_decay=1e-4,name="",ki=initializers.RandomNormal(mean=0.0,stddev=0.02),kr=None):

    kr = regularizers.l2(weight_decay)

    x = Conv2D(params.FILTER, (ks, ks), padding='same', kernel_initializer=ki, kernel_regularizer=kr,
               activation='linear',  use_bias=True, strides=(stride, stride), name='last_layer'+name)(x)

    x = Reshape((params.GRID_H, params.GRID_W, params.BOX, params.CLASS + 4 + 1), name='yolo_head'+name)(x)

    return x

def conv_block(x, channels, kernel_size=3, stride=1, weight_decay=5e-4, dropout_rate=None,act='l'):
    kr = regularizers.l2(weight_decay)
    ki = initializers.RandomNormal(mean=0.0,stddev=0.02)


    x = Conv2D(channels, (kernel_size, kernel_size), kernel_initializer=ki, strides=(stride, stride),
               use_bias=False, padding='same', kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)

    if(act == 'm'):
        x = Mish()(x)

    if (act == 'l'):
        x = LeakyReLU(alpha=0.1)(x)

    if (act == 'r'):
        x = Activation('relu')(x)

    if (act == 's'):
        x = Activation('sigmoid')(x)

    if dropout_rate != None and dropout_rate != 0.:
        x = Dropout(dropout_rate)(x)

    return x


def separable_conv_block(x, channels, kernel_size=3, stride=1, weight_decay=5e-4, dropout_rate=None,act='l'):
    ki = initializers.RandomNormal(mean=0.0, stddev=0.2)

    kr = regularizers.l2(weight_decay)

    x = SeparableConv2D(channels, (kernel_size, kernel_size), kernel_initializer=ki,
                        strides=(stride, stride), use_bias=False, padding='same',
                        kernel_regularizer=kr)(x)

    x = BatchNormalization()(x)

    if (act == 'm'):
        x = Mish()(x)

    if (act == 'l'):
        x = LeakyReLU(alpha=0.1)(x)

    if (act == 'r'):
        x = Activation('relu')(x)

    if (act == 's'):
        x = Activation('sigmoid')(x)

    if dropout_rate != None and dropout_rate != 0.:
        x = SpatialDropout2D(dropout_rate)(x)
    return x

def atrous_block(x, channels, kernel_size=3, stride=1, weight_decay=5e-4, dropout_rate=None):
    #From
    #Christos Kyrkou and Theocharis Theocharides, "EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring Using Atrous Convolutional Feature Fusion," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 1687-1699, 2020, doi: 10.1109/JSTARS.2020.2969809.

    ki = initializers.he_normal()
    kr = regularizers.l2(weight_decay)


    x1 = DepthwiseConv2D(kernel_size=3, kernel_regularizer=kr, kernel_initializer=ki, strides=1, padding='same',
                        use_bias=False, dilation_rate=1)(x)
    x1 = BatchNormalization()(x1)
    x1 = Mish()(x1)
    x2 = DepthwiseConv2D(kernel_size=3, kernel_regularizer=kr, kernel_initializer=ki, strides=1, padding='same',
                         use_bias=False, dilation_rate=2)(x)
    x2 = BatchNormalization()(x2)
    x2 = Mish()(x2)
    x3 = DepthwiseConv2D(kernel_size=3, kernel_regularizer=kr, kernel_initializer=ki, strides=1, padding='same',
                         use_bias=False, dilation_rate=3)(x)
    x3 = BatchNormalization()(x3)
    x3 = Mish()(x3)

    x = tf.keras.layers.Maximum()([x1,x2,x3])


    x = Conv2D(channels, (1, 1), kernel_regularizer=kr, kernel_initializer=ki, strides=(1, 1), padding='same', use_bias=False)(x)

    if dropout_rate != None and dropout_rate != 0.:
        x = SpatialDropout2D(dropout_rate)(x)
    return x

def squeeze_excite_block(tensor, ratio=16):

    dim = K.int_shape(tensor)[-1]
    se_shape = (1, 1, dim)

    se = GlobalAveragePooling2D()(tensor)
    se = Reshape(se_shape)(se)
    se = Dense(dim // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(dim, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([tensor, se])
    return x

def spatial_attention_block(tensor):

    x1 = K.max(tensor,axis=-1,keepdims=True)
    x2 = K.mean(tensor,axis=-1,keepdims=True)

    x = Concatenate(axis=-1)([x1,x2])
    x = Conv2D(1, (3, 3), use_bias=True, padding='same')(x)
    x = Activation('sigmoid')(x)

    x = Multiply()([tensor, x])

    return x

def enc_resnet(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False, pooling=None)
    base_model.summary()
    istrainable = True

    for ind, layer in enumerate(base_model.layers):
        print(ind, layer.name)
        if (layer.name == 'conv5_block3_out'):
            layer.trainable = istrainable
            x = base_model.layers[ind].output
            break
        if (layer.name == 'conv4_block6_out'): #Freeze the whole backbone
            istrainable = True
        layer.trainable = istrainable


    x = BatchNormalization()(x)

    x = conv_block(x, 256, kernel_size=1, stride=1, dropout_rate=0.25,act='r')


    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, base_model.input


def enc_resnet25(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False, pooling=None)
    istrainable = True

    base_model.summary()

    for ind, layer in enumerate(base_model.layers):
        print(ind, layer.name)
        if (layer.name == 'activation_25'):
            layer.trainable = istrainable
            x = base_model.layers[ind].output
            break
        if (layer.name == 'activation_11'):
            istrainable = True
        layer.trainable = istrainable

    cls = x
    x = separable_conv_block(x, 256, kernel_size=2, stride=2, dropout_rate=0.)
    x = separable_conv_block(x, 256, kernel_size=1, stride=1, dropout_rate=0.)

    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, base_model.input, cls


def enc_vggnet(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    base_model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False, pooling=None)
    base_model.summary()

    istrainable = False

    for ind, layer in enumerate(base_model.layers):
        print(ind, layer.name)
        if (layer.name == 'block5_pool'):
            layer.trainable = istrainable
            x = base_model.layers[ind].output
            break
        if (layer.name == 'block5_conv3'):
            istrainable = True
        layer.trainable = istrainable

    cls = x

    x = conv_block(x, 256, kernel_size=3, stride=1, dropout_rate=0.25)

    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, base_model.input, cls


def enc_mobile(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]

    base_model = MobileNet(input_shape, weights='imagenet', include_top=False, pooling=None)
    base_model.summary()

    istrainable = True

    for ind, layer in enumerate(base_model.layers):
        if (layer.name == 'conv_pw_13_relu'):
            layer.trainable = istrainable
            x = base_model.layers[ind].output
            break

        if (layer.name == 'conv_pw_11_relu'):
            istrainable = True
        layer.trainable = istrainable

    x = SpatialDropout2D(0.25)(x)
    x = conv_block(x, 256, kernel_size=1, stride=1, dropout_rate=0.,weight_decay=0.)

    x = add_YOLO_Head(params, x, ks=1, stride=1,weight_decay=0., ki=initializers.he_normal())

    return x, base_model.input


def enc_mobileV2(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    base_model = MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False, pooling=None)

    base_model.summary()
    istrainable = True

    for ind, layer in enumerate(base_model.layers):
        if (layer.name == 'out_relu'):
            layer.trainable = istrainable
            x = base_model.layers[ind].output
            break
        if (layer.name == 'block_14_add'):
            istrainable = True
        layer.trainable = istrainable


    x = Dropout(0.4)(x)

    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, base_model.input


def enc_EfficientNetB0(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    base_model = EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=False, pooling=None)

    base_model.summary()
    istrainable = True

    for ind, layer in enumerate(base_model.layers):
        if (layer.name == 'out_relu'):
            layer.trainable = istrainable
            x = base_model.layers[ind].output
            break
        if (layer.name == 'block_12_add'):
            istrainable = True
        layer.trainable = istrainable

    x = SpatialDropout2D(0.25)(x)

    x = separable_conv_block(x, 512, kernel_size=1, stride=1, dropout_rate=0.25)

    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, base_model.input


def enc_nasmobile(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    base_model = NASNetMobile(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling=None)
    base_model = change_model(base_model, new_input_shape=[None] + input_shape, custom_objects={})

    base_model.summary()
    istrainable = True

    for ind, layer in enumerate(base_model.layers):
        if (layer.name == 'out_relu'):
            layer.trainable = istrainable
            x = base_model.layers[ind].output
            break
        if (layer.name == 'block_12_add'):
            istrainable = True
        layer.trainable = istrainable

    x = base_model.layers[-1].output

    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, base_model.input


def enc_yolopeds(params,name=''):

    input_shape = [params.NORM_H, params.NORM_W, 3]
    inp = Input(shape=input_shape)


    x = conv_block(inp, 16, 5, stride=2, dropout_rate=0.)#conv
    x = separable_conv_block(x, 32, 5, dropout_rate=0.)#conv1

    x = separable_conv_block(x, 32, 5, dropout_rate=0.)#conv2
    x = separable_conv_block(x, 32, 5, dropout_rate=0)#conv3
    x1 = AveragePooling2D(pool_size=(2, 2))(x)  # 80x80

    x2 = separable_conv_block(x1, 64, 3, dropout_rate=0.)#conv4
    x = separable_conv_block(x2, 128, 3, dropout_rate=0.)#conv5
    x = Concatenate()([x, x1])

    x3 = AveragePooling2D(pool_size=(2, 2))(x)  # 40x40

    x4 = separable_conv_block(x3, 128, 3, dropout_rate=0.)#conv6
    x = separable_conv_block(x4, 128, 3, dropout_rate=0.)#conv7
    x = Concatenate()([x, x3])

    x5 = AveragePooling2D(pool_size=(2, 2))(x)  # 20x20

    x = separable_conv_block(x5, 256, 3, dropout_rate=0.)#conv8
    x = separable_conv_block(x, 256, 3, dropout_rate=0.)#conv9
    x = Concatenate()([x, x5])

    x = AveragePooling2D(pool_size=(2, 2))(x)  # 10x10
    x = spatial_attention_block(x)

    xb = AveragePooling2D(pool_size=(8, 8))(x2)
    xb = spatial_attention_block(xb)

    xc = AveragePooling2D(pool_size=(4, 4))(x4)
    xc = spatial_attention_block(xc)

    x = Concatenate()([x,
                       xb,
                       xc])

    cls = x

    custom_objects = {'Mish': Mish}

    x = add_YOLO_Head(params, x, ks=1, stride=1,name="")

    return x, inp, cls

def enc_dronet(params,name=''):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    inp = Input(shape=input_shape)

    act='l'
    block = conv_block
    x = block(inp, 32, 5, stride=2, dropout_rate=0.,act=act)

    x = block(x, 32, 5, stride=1, dropout_rate=0.,act=act)
    x = block(x, 32, 1, stride=2, dropout_rate=0.,act=act)


    x = block(x, 64, 3, stride=1, dropout_rate=0.,act=act)
    x = block(x, 64, 1, stride=1, dropout_rate=0.,act=act)
    x = MaxPooling2D(pool_size=(2, 2))(x)


    x = block(x, 64, 3, stride=1, dropout_rate=0.,act=act)
    x = block(x, 64, 1, stride=1, dropout_rate=0.,act=act)
    x = MaxPooling2D(pool_size=(2, 2))(x)


    x4 = block(x, 64, 3, stride=2, dropout_rate=0.,act=act)
    x2 = block(x4, 128, 1, stride=1, dropout_rate=0.,act=act)


    x = block(x2, 128, 3, stride=1, dropout_rate=0., act=act)
    x = block(x, 256, 1, stride=1, dropout_rate=0., act=act)

    x = block(x, 128, 3, stride=1, dropout_rate=0.,act=act)
    x3 = block(x, 512, 1, stride=1, dropout_rate=0.,act=act)

    x=x3


    cls = x

    custom_objects = {'Mish': Mish}

    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, inp, cls

def swish(x):
    def hard_swish(x):
        hrelu = tf.nn.relu6(tf.add(x, tf.constant(3.0)))
        tf.multiply(x, hrelu)
        return x

    return hard_swish(x)

def enc_tinyyolov2(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    inp = Input(shape=input_shape)

    kr = regularizers.l2(5e-4)
    ki = initializers.he_normal()#initializers.RandomNormal(mean=0.0,stddev=0.01)
    # Layer 1
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False, input_shape=(416, 416, 3),
               kernel_regularizer=kr,
               kernel_initializer=ki)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2 - 5
    for i in range(0, 4):
        x = Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=kr,
                   kernel_initializer=ki)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = SpatialDropout2D(0.25)(x)

    # Layer 6
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=kr,
               kernel_initializer=ki)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Layer 7 - 8
    for _ in range(0, 2):
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=kr,
                   kernel_initializer=ki)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = SpatialDropout2D(0.25)(x)

    # Layer 9
    cls = x
    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, inp, cls


class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def enc_miniGoogleNet(params):
    input_shape = [params.NORM_H, params.NORM_W, 3]

    mgn = MiniGoogLeNet()

    model = mgn.build2(params.NORM_W, params.NORM_H, 3, 3)

    model.summary()

    inp = model.input
    x = model.output
    x = AveragePooling2D()(x)
    cls = x
    # x = Dropout(0.3)(x)

    x = add_YOLO_Head(params, x, ks=1, stride=1)

    return x, inp, cls

def MYMODEL(params, net_name='resnet'):
    input_shape = [params.NORM_H, params.NORM_W, 3]
    inputs = Input(shape=input_shape)

    cls = None

    if (net_name == 'resnet'):
        x, inp = enc_resnet(params)

    if (net_name == 'resnet25'):
        x, inp, _ = enc_resnet25(params)

    if (net_name == 'efficient'):
        x, inp = enc_EfficientNetB0(params)

    if (net_name == 'nasmobile'):
        x, inp = enc_nasmobile(params)

    if (net_name == 'mobileV2'):
        x, inp = enc_mobileV2(params)

    if (net_name == 'mobile'):
        x, inp = enc_mobile(params)

    if (net_name == 'vgg'):
        x, inp, _ = enc_vggnet(params)

    if (net_name == 'miniGoogleNet'):
        x, inp, _ = enc_miniGoogleNet(params)

    if (net_name == 'yolopeds'):
        x, inp, _ = enc_yolopeds(params,name=net_name)

    if (net_name == 'dense'):
        x, inp, _ = enc_dense(params)

    if (net_name == 'dronet'):
        x, inp, _ = enc_dronet(params,name=net_name)

    if (net_name == 'tinyyolov2'):
        x, inp, _ = enc_tinyyolov2(params)


    if (cls == None):
        m = Model(inputs=[inp], outputs=[x])

    else:
        cls = conv_block(cls, channels=params.CLASS, kernel_size=1, dropout_rate=0.25)
        cls = GlobalAveragePooling2D()(cls)

        cls = Activation('sigmoid', name='class_branch')(cls)
        m = Model(inputs=[inp], outputs=[x, cls])

    return m
