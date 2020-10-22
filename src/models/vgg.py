
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D

from src.layers.conv import VggBlock


class VGG8(Model):

    def __init__(self):
        super(VGG8, self).__init__()
        weight_decay = 1e-4
        self.vgg_1 = VggBlock(16)
        self.vgg_2 = VggBlock(32)
        self.vgg_3 = VggBlock(64)
        self.vgg_4 = VggBlock(128)
        self.max_pool = MaxPool2D(pool_size=(2, 2))
        self.drop_out = Dropout(0.3)
        self.gap = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.dense_1 = Dense(10, kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.dense_2 = Dense(10, activation='softmax',
                             kernel_regularizer=keras.regularizers.l2(weight_decay))

    def call(self, inputs):
        x = self.vgg_1(inputs)
        x = self.max_pool(x)
        x = self.vgg_2(x)
        x = self.max_pool(x)
        x = self.vgg_3(x)
        x = self.max_pool(x)
        x = self.vgg_4(x)
        x = self.max_pool(x)
        x = self.batch_norm_1(x)
        x = self.drop_out(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        output = self.batch_norm_2(x)
        # output = self.dense_2(x)

        return output
