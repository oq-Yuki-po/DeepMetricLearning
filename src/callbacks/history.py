import keras
from keras.models import Model
import numpy as np

from config import Config
from src.visualize import plot_embedde_space, plot_learning_hitory


class Histories(keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):

        plot_learning_hitory(self.train_loss, self.val_loss, self.train_acc, self.val_acc)

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.train_acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])

        model = self.model
        model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
        features = model.predict(self.validation_data[0], verbose=0)
        features /= np.linalg.norm(features, axis=1, keepdims=True)

        # plot
        plot_embedde_space(features, self.validation_data[1], f'{Config.EMBEDDED_PATH}/epoch_{epoch+1}.png')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
