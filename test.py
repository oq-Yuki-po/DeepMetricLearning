import argparse
import math
import os
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from glob import glob

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import (CSVLogger, EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.datasets import mnist
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, StratifiedKFold


def main():
    # dataset
    (X, y), (X_test, y_test) = mnist.load_data()

    X = X[:, :, :, np.newaxis].astype('float32') / 255
    X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255
    y_ohe = keras.utils.to_categorical(y, 10)
    y_ohe_test = keras.utils.to_categorical(y_test, 10)

    # feature extraction
    arcface_model = load_model('saved_model/arcface')
    arcface_model = Model(inputs=arcface_model.input[0], outputs=arcface_model.layers[-3].output)
    arcface_features = arcface_model.predict(X_test, verbose=1)
    arcface_features /= np.linalg.norm(arcface_features, axis=1, keepdims=True)

    # plot

    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    for c in range(len(np.unique(y_test))):
        ax2.plot(arcface_features[y_test==c, 0], arcface_features[y_test==c, 1], arcface_features[y_test==c, 2], '.', alpha=0.1)
    plt.title('ArcFace')


    plt.savefig('figure.png')


if __name__ == '__main__':
    main()
