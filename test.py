import keras
from keras.datasets import mnist
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from config import Config
from src.data.mnist import load_mnist


def main():
    # dataset
    (_, _), (X_test, _), y_test = load_mnist()

    # feature extraction
    model = load_model(Config.MODEL_PATH)
    model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
    features = model.predict(X_test, verbose=0)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    # plot
    fig = plt.figure()
    ax = Axes3D(fig)
    for c in range(len(np.unique(y_test))):
        ax.plot(features[y_test == c, 0],
                features[y_test == c, 1],
                features[y_test == c, 2],
                '.',
                alpha=0.5)

    plt.title(Config.MODEL_NAME)

    plt.savefig(Config.TEST_RESULT)


if __name__ == '__main__':
    main()
