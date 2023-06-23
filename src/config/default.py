from __future__ import absolute_import, division, print_function

from yacs.config import CfgNode as CN

_C = CN()

_C.CFG_PATH = ""
_C.TASK = "simclr_pretrain"
_C.SAMPLE_METHOD = "coco_hp"
_C.DATA_DIR = "/data"
_C.EXP_ID = "default"
_C.DEBUG = 0
_C.DEBUG_THEME = "white"
_C.TEST = False
_C.SEED = 317
_C.SAVE_RESULTS = False

_C.OUTPUT_DIR = ""
_C.LOG_DIR = ""
_C.EXPERIMENT_NAME = ""
_C.GPUS = [0, 1, 2, 3]
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.ENABLED = True
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.PRETRAINED = ""
_C.MODEL.INIT_WEIGHTS = False
_C.MODEL.NAME = "res_50"
_C.MODEL.HEAD_NAME = "projection"
_C.MODEL.FEATURE_DIM = 64
_C.MODEL.NUM_CLASSES = 1
_C.MODEL.INPUT_W = 224
_C.MODEL.INPUT_H = 224
# mae params
_C.MODEL.MAE_EMBED_DIM = 192
_C.MODEL.MAE_PATCH_SIZE = 2
_C.MODEL.MAE_ENCODER_DEPTH = 12
_C.MODEL.MAE_ENCODER_NUM_HEADS = 6
_C.MODEL.MAE_DECODER_DEPTH = 4
_C.MODEL.MAE_DECODER_NUM_HEADS = 3
_C.MODEL.MAE_MASK_RATIO = 0.75

_C.LOSS = CN()
_C.LOSS.METRIC = ["loss"]
_C.LOSS.TEMPERATURE = 0.5
# infonce loss
_C.LOSS.INFONCE_GAMMA = 1.0
# sinkhorn loss
_C.LOSS.SINKHORN_MAX_ITER = 5
_C.LOSS.SINKHORN_GAMMA = 1.0
# gromov wasserstein loss
_C.LOSS.GW_MAX_ITER = 5
_C.LOSS.GW_EPSILON = 0.05
_C.LOSS.GW_GAMMA = 1.0
# mae mse loss
_C.LOSS.NORM_PIX_LOSS = False
_C.LOSS.NORM_PIX_LOSS_EPSILON = 1e-6

# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATASET = "coco_hp"
_C.DATASET.IMAGE_SIZE = 224
_C.DATASET.TRAIN_SET = "train"
_C.DATASET.TEST_SET = "valid"
_C.DATASET.TRAIN_IMAGE_DIR = "images/train2017"
_C.DATASET.TRAIN_ANNOTATIONS = ["person_keypoints_train2017.json"]
_C.DATASET.VAL_IMAGE_DIR = "images/val2017"
_C.DATASET.VAL_ANNOTATIONS = "person_keypoints_val2017.json"
# training data augmentation
_C.DATASET.USE_TEST_AUG = False
_C.DATASET.MEAN = [0.408, 0.447, 0.470]
_C.DATASET.STD = [0.289, 0.274, 0.278]
_C.DATASET.RANDOM_CROP = True
_C.DATASET.RANDOM_RESIZED_CROP = 32
_C.DATASET.RANDOM_HORIZONTAL_FLIP = 0.5
# color jitter
_C.DATASET.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.8]
_C.DATASET.RANDOM_GRAYSCALE = 0.2
_C.DATASET.SHIFT = 0.1
_C.DATASET.SCALE = 0.4
_C.DATASET.ROTATE = 0.0


# train
_C.TRAIN = CN()

_C.TRAIN.LR_SCHEDULER = "linear"
_C.TRAIN.DISTRIBUTE = True
_C.TRAIN.LOCAL_RANK = 0
_C.TRAIN.HIDE_DATA_TIME = False
_C.TRAIN.SAVE_ALL_MODEL = False
_C.TRAIN.RESUME = False
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 120]
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WARMUP_LR = 1e-5
_C.TRAIN.BASE_LR = 1e-3
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.MIN_LR = 1e-5
_C.TRAIN.EPOCHS = 140
_C.TRAIN.NUM_ITERS = -1
_C.TRAIN.LR = 1.25e-4
_C.TRAIN.WD = 1e-6
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.MASTER_BATCH_SIZE = -1


_C.TRAIN.OPTIMIZER = "adam"
_C.TRAIN.MOMENTUM = 0.9

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ""
_C.TRAIN.SHUFFLE = True
_C.TRAIN.VAL_INTERVALS = 5
_C.TRAIN.TRAINVAL = False

# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 256
_C.TEST.OUTPUT_FILE = "predictions.csv"
# Test Model Epoch
_C.TEST.TASK = ""
_C.TEST.MODEL_PATH = ""


def update_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(_C, file=f)
