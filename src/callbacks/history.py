import sys

import keras
from keras.models import Model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import roc_auc_score

from config import Config


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

        plt.figure(num=1, clear=True)
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(self.train_acc, label='train')
        plt.plot(self.val_acc, label='validation')
        plt.legend()
        plt.savefig(f'{Config.HISTORRY_PATH}/accuracy.png')

        plt.figure(num=1, clear=True)
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(self.train_loss, label='train')
        plt.plot(self.val_loss, label='validation')
        plt.legend()
        plt.savefig(f'{Config.HISTORRY_PATH}/loss.png')

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
        fig = plt.figure()
        ax = Axes3D(fig)
        for c in range(len(np.unique(self.validation_data[1]))):
            ax.plot(features[self.validation_data[1] == c, 0],
                    features[self.validation_data[1] == c, 1],
                    features[self.validation_data[1] == c, 2],
                    '.',
                    alpha=0.5)
        plt.title('ArcFace')

        plt.savefig(f'{Config.EMBEDDED_PATH}/{epoch}.png')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
