from keras.models import Model, load_model
import numpy as np

from config import Config
from src.data.mnist import load_mnist
from src.visualize import plot_embedde_space


def main():
    # dataset
    (_, _), (X_test, y_test_categorical) = load_mnist()

    # feature extraction
    model = load_model(Config.MODEL_PATH)
    model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
    features = model.predict(X_test, verbose=0)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    # plot
    plot_embedde_space(features, np.argmax(y_test_categorical, axis=1), Config.TEST_RESULT)


if __name__ == '__main__':
    main()
