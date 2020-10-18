from tensorflow import keras
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Layer)
from tensorflow.keras.layers.experimental.preprocessing import Resizing


class ResizeLayer(Layer):
    def __init__(self):
        super(ResizeLayer, self).__init__()
        self.resize = Resizing(20, 20, interpolation='bilinear')

    def call(self, inputs):

        output = self.resize(inputs)
        
        return output


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size=(3, 3)):
        super(ConvBlock, self).__init__()

        weight_decay = 1e-4
        self.conv_2d = Conv2D(filters,
                              kernel_size,
                              padding='same',
                              kernel_initializer='he_normal',
                              kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.batch_norm = BatchNormalization()
        self.activation = Activation('relu')

    def call(self, inputs, is_batch_norm=True):
        x = self.conv_2d(inputs)
        if is_batch_norm:
            x = self.batch_norm(x)
        output = self.activation(x)
        return output

class VggBlock(Layer):
    def __init__(self, filters, kernel_size=(3, 3)):
        super(VggBlock, self).__init__()

        self.conv_1 = ConvBlock(filters, kernel_size)
        self.conv_2 = ConvBlock(filters, kernel_size)

    def call(self, inputs):

        x = self.conv_1(inputs)
        output = self.conv_2(x)

        return output
