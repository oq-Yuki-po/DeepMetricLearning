import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist


def load_mnist():
    '''
    mnistの前処理を含めたデータロード
    '''
    (X, y), (X_test, y_test) = mnist.load_data()

    # 前処理
    X = X[:, :, :, np.newaxis].astype('float32') / 255
    X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255
    y_categorical = keras.utils.to_categorical(y, 10)
    y_test_categorical = keras.utils.to_categorical(y_test, 10)

    return (X, y_categorical), (X_test, y_test_categorical), y_test
