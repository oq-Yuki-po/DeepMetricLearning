class Config:
    IMAGE_SHAPE = (32, 32, 3)
    NUM_CLASSES = 10
    MODEL_NAME = 'monilenet_v2'
    MODEL_PATH = f'results/saved_model/{MODEL_NAME}'
    EPOCH = 5
    BATCH_SIZE = 512
    CHECKPOINT_PATH = 'results/checkpoints/cp-{epoch:04d}.ckpt'
    EMBEDDED_PATH = 'results/embedded'
    HISTORRY_PATH = 'results/history'
    TEST_RESULT = 'test/result.png'
