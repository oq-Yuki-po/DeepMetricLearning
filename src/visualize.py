import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from config import Config


def plot_embedde_space(features, labels, name):
    '''
    埋め込み空間の描画
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    for c in range(len(np.unique(labels))):
        ax.plot(features[labels == c, 0],
                features[labels == c, 1],
                features[labels == c, 2],
                '.',
                alpha=0.5)
    plt.title(name)

    plt.savefig(name)


def plot_learning_hitory(train_loss, val_loss, train_acc, val_acc):
    '''
    学習履歴の描画
    '''
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))

    titles = ['accuracy', 'loss']

    for i in range(0, 2):
        plt.subplot(2, 1, i + 1)
        plt.title(titles[i])
        plt.xlabel('epoch')
        plt.ylabel(titles[i])
        if i == 0:
            plt.plot(train_acc, label='train')
            plt.plot(val_acc, label='validation')
        else:
            plt.plot(train_loss, label='train')
            plt.plot(val_loss, label='validation')
        plt.tight_layout()
        plt.legend()

    plt.savefig(f'{Config.HISTORRY_PATH}/history.png')
