class Config:
    IMAGE_SHAPE = (28, 28, 1)
    NUM_CLASSES = 10
    MODEL_PATH = 'saved_model/arcface'
    EPOCH = 50
    BATCH_SIZE = 512
    CHECKPOINT_PATH = 'results/checkpoints/cp-{epoch:04d}.ckpt'
    EMBEDDED_PATH = 'results/embedded'
    HISTORRY_PATH = 'results/history'
