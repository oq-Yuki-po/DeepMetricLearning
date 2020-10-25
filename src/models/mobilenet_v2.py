from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class BaseMobileNetV2(Model):

    def __init__(self, input_shape):
        super(BaseMobileNetV2, self).__init__()
        weight_decay = 1e-4
        self.base_model = MobileNetV2(input_shape=input_shape,
                                      include_top=False,
                                      weights='imagenet')
        self.base_model.trainable = False
        self.gap = GlobalAveragePooling2D()
        self.dense = Dense(10,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(weight_decay))

    def call(self, inputs):

        x = self.base_model(inputs)
        x = self.gap(x)
        output = self.dense(x)

        return output
