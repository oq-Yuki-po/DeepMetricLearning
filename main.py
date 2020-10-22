import numpy as np
from tensorflow import keras
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD

from config import Config
from src.metrics import ArcFace
from src.models.vgg import VGG8


def main(is_model_loaded=False):

    if is_model_loaded is False:
        # 入力層の定義
        # 画像の形
        inputs = Input(shape=Config.IMAGE_SHAPE)
        # 分類クラス数
        y = Input(shape=(Config.NUM_CLASSES,))

        weight_decay = 1e-4

        # 特徴抽出のモデルを構築
        base_model = VGG8()
        base_model_out = base_model(inputs)

        # 距離を計測するレイヤー作成
        output = ArcFace(n_classes=Config.NUM_CLASSES,
                         regularizer=regularizers.l2(weight_decay))([base_model_out, y])

        # モデルの構築
        model = Model([inputs, y], output)

        # モデルのコンパイル
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=1e-3, momentum=0.5),
                      metrics=['accuracy'])
    else:
        model = keras.models.load_model('saved_model/arcface')

    # model.summary()

    # データセットの用意
    (X, y), (X_test, y_test) = mnist.load_data()

    # 前処理
    X = X[:, :, :, np.newaxis].astype('float32') / 255
    X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255
    y = keras.utils.to_categorical(y, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # callbacks
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=Config.CHECKPOINT_PATH,
                                                  verbose=1,
                                                  save_weights_only=True,
                                                  period=5)


    # 学習
    model.fit(x=[X, y],
              y=y,
              validation_data=([X_test, y_test], y_test),
              batch_size=Config.BATCH_SIZE,
              epochs=Config.EPOCH,
              callbacks=[cp_callback],
              verbose=1)

    # モデルの保存
    model.save(Config.MODEL_PATH)


if __name__ == '__main__':
    main()
