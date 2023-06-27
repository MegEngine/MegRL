#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from basecore.config import ConfigDict

__all__ = [
    "DataConfig",
    "ModelConfig",
    # "CityscapesConfig",
    "SolverConfig",
    "TrainerConfig",
    "TestConfig",
    "GlobalConfig",
]


"""
class CityscapesConfig(DataConfig):

    def __init__(self):
        super().__init__()
        self.TRAIN = dict(
            name="cityscapes_train",
            image_set="train",
            mode="gtFine",
            order=("image", "mask", "info"),
        )
        self.TEST = dict(
            name="cityscapes_val",
            image_set="val",
            mode="gtFine",
            order=("image", "mask", "info"),
        )
        self.NUM_CLASSES = 19
"""


class DataConfig(ConfigDict):

    def __init__(self):
        self.TASK_TYPE = ''
        self.TASK_NAME = ''
        self.TASK_NAME_ENVPOOL = ''
        self.TRAINING_ENV_ARGS = dict()
        self.TEST_ENV_ARGS = dict()
        self.TRAINING_ENV_NUM = 10
        self.TEST_ENV_NUM = 10
        self.BUFFER_NAME = ''
        self.BUFFER_SIZE = None


class GlobalConfig(ConfigDict):

    def __init__(self):
        self.OUTPUT_DIR = ""
        self.CKPT_SAVE_DIR = ""
        # # use the following ckpt_save_dir for oss user
        self.TENSORBOARD = dict(
            ENABLE=False,
        )
        self.LOG_INTERVAL = None


class ModelConfig(ConfigDict):

    def __init__(self):
        self.BACKBONE = dict(
            NAME='',
            TYPE='',
            CONFIGURATIONS={}
        )
        self.STATE_SHAPE = None
        self.ACTION_SHAPE = None
        
        self.POLICY_NAME = ''
        self.POLICY_TRAINER_NAME = ''
        self.GAMMA = None
        self.N_STEP = None
        self.TARGET_FREQ = None 
        self.TEST_EPS = None 
        self.EPS_BEGIN = None
        self.EPS_END = None
        self.EPS_SCHEDULE = "" 
        self.EPS_SCHEDULE_LENGTH = None
        self.COLLECT_BEFORE_TRAIN = None
        self.STEP_PER_COLLECT = None
        self.UPDATE_PER_STEP = None # train_step in each epoch = update_per_step * NUM_ENV_STEP_PER_EPOCH
        self.REWARD_METRIC = None
        
        self.BATCHSIZE_PER_DEVICE = 1 # useless, for solver (lr = cfg.SOLVER.BASIC_LR * cfg.MODEL.BATCHSIZE_PER_DEVICE)
        self.WEIGHTS = None # for resume
        # self.BACKBONE = dict(
        #     NAME="resnet50",
        #     IMG_MEAN=[103.530, 116.280, 123.675],  # BGR
        #     IMG_STD=[57.375, 57.12, 58.395],
        #     NORM="SyncBN",
        #     FREEZE_AT=0,
        # )


class SolverConfig(ConfigDict):

    def __init__(self):
        self.OPTIMIZER_NAME = "Adam"
        self.BASIC_LR = None
        self.WEIGHT_DECAY = 1e-4
        self.EXTRA_OPT_ARGS = dict(
            # momentum=0.9,
        )
        self.REDUCE_MODE = "MEAN"
        self.WARM_ITERS = 0
        self.LR_SCHEDULE = None
        # self.NUM_IMAGE_PER_EPOCH = None
        self.MAX_EPOCH = None
        # self.LR_DECAY_STAGES = [200, 400]
        # self.LR_DECAY_RATE = 0.1
        
        self.NUM_ENV_STEP_PER_EPOCH = None


class TrainerConfig(ConfigDict):

    def __init__(self):
        self.RESUME = False
        self.AMP = dict(
            ENABLE=False,
            # when dynamic scale is enabled, we start with a higher scale of 65536,
            # scale is doubled every 2000 iter or halved once inf is detected during training.
            DYNAMIC_SCALE=False,
        )
        self.EMA = dict(
            ENABLE=False,
            # ALPHA=5e-4,
            # MOMENTUM=None,
            # UPDATE_PERIOD=1,
            # BURNIN_ITER=2000,
        )
        self.GRAD_CLIP = dict(
            ENABLE=False,
            # supported type: ("value", "norm")
            # TYPE="value",
            # ARGS=dict(lower=-1, upper=1)
            # ARGS=dict(max_norm=1.0, ord=2)
        )
        
        self.BATCH_SIZE = None
        

class TestConfig(ConfigDict):

    def __init__(self):
        self.EVAL_EPOCH_INTERVAL = 1
