from tensorflow import keras
from tensorflow.keras.datasets import cifar10

from config import Config


def load_cifar10():
    '''
    cifar10の前処理を含めたデータロード
    '''
    (X, y), (X_test, y_test) = cifar10.load_data()

    # 前処理
    y_categorical = keras.utils.to_categorical(y, Config.NUM_CLASSES)
    y_test_categorical = keras.utils.to_categorical(y_test, Config.NUM_CLASSES)

    return (X, y_categorical), (X_test, y_test_categorical)
