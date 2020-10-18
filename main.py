import numpy as np
from tensorflow import keras
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD, Adam

from layers.arcface import ArcFace
from vgg import VGG8

inputs = Input(shape=(28, 28, 1))
y = Input(shape=(10,))

weight_decay = 1e-4

base_model = VGG8()

base_model_out = base_model(inputs)

output = ArcFace(10,
                 regularizer=regularizers.l2(weight_decay))([base_model_out, y])

model = Model([inputs, y], output)

model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr=1e-3, momentum=0.5),
              metrics=['accuracy'])

model.summary()

(X, y), (X_test, y_test) = mnist.load_data()

X = X[:, :, :, np.newaxis].astype('float32') / 255
X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255

y = keras.utils.to_categorical(y, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# model = keras.models.load_model('saved_model/arcface')

model.fit([X, y], y, validation_data=([X_test, y_test], y_test),
          batch_size=512,
          epochs=30,
          verbose=1)

model.save('saved_model/arcface')
