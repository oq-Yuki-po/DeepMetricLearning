import keras
from keras.datasets import mnist
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def main():
    # dataset
    (X, y), (X_test, y_test) = mnist.load_data()

    X = X[:, :, :, np.newaxis].astype('float32') / 255
    X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255

    # feature extraction
    arcface_model = load_model('saved_model/arcface')
    arcface_model = Model(inputs=arcface_model.input[0], outputs=arcface_model.layers[-3].output)
    arcface_features = arcface_model.predict(X_test, verbose=1)
    arcface_features /= np.linalg.norm(arcface_features, axis=1, keepdims=True)

    # plot
    fig = plt.figure()
    ax = Axes3D(fig)
    for c in range(len(np.unique(y_test))):
        ax.plot(arcface_features[y_test == c, 0],
                arcface_features[y_test == c, 1],
                arcface_features[y_test == c, 2],
                '.',
                alpha=0.5)
    plt.title('ArcFace')

    plt.savefig('figure.png')


if __name__ == '__main__':
    main()
