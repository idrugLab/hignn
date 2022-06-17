# -*- coding: utf-8 -*-
"""
@Author  : Weimin Zhu
@Time    : 2021-09-28
@File    : config.py
"""

import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------
# Path to output folder
_C.OUTPUT_DIR = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Fixed random seed
_C.SEED = 1
# Number of folds to run
_C.NUM_FOLDS = 10
# Whether to show individual scores for each task
_C.SHOW_EACH_SCORES = False
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Frequency to show training epoch
_C.SHOW_FREQ = 5

# Hyperopt setting
_C.HYPER = False
_C.HYPER_COUNT = 1
_C.HYPER_REMOVE = None
# Number of hyperparameters choice to try
_C.NUM_ITERS = 20

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size, overwritten by command line argument
_C.DATA.BATCH_SIZE = 64
# Path to dataset, overwritten by command line argument
_C.DATA.DATA_PATH = '../data/'
# Dataset name
_C.DATA.DATASET = 'bace'
# Tasks name, override by ~get_task_names~(utlis.py 152) function
_C.DATA.TASK_NAME = None
# Dataset type, 'classification' or 'regression'
_C.DATA.TASK_TYPE = 'classification'
# Metric, choice from ['auc', 'prc', 'rmse', 'mae']
_C.DATA.METRIC = 'auc'
# How to split data, 'random', 'scaffold' or 'noise'
_C.DATA.SPLIT_TYPE = 'random'
# anti-noise rate for hiv dataset, only works when DATA.SPLIT_TYPE is 'noise'
_C.DATA.RATE = None

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Hidden size of HiGNN model
_C.MODEL.HID = 64
# Output size of HiGNN model, override by dataset.py 474
_C.MODEL.OUT_DIM = None
# Number of layers
_C.MODEL.DEPTH = 3
# Number of heads
_C.MODEL.SLICES = 2
# Dropout
_C.MODEL.DROPOUT = 0.2
# Feature attention
_C.MODEL.F_ATT = True
# reduction value
_C.MODEL.R = 4
# Whether to use BRICS information, if set to False, the option LOSS.CL_LOSS is set to False
_C.MODEL.BRICS = True

# -----------------------------------------------------------------------------
# Loss settings
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Whether to adopt focal loss
_C.LOSS.FL_LOSS = False
# Whether to adopt molecule-fragment contrastive learning
_C.LOSS.CL_LOSS = False
# Alpha
_C.LOSS.ALPHA = 0.1
# Scale logits by the inverse of the temperature
_C.LOSS.TEMPERATURE = 0.1

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Checkpoint to resume, overwritten by command line argument
_C.TRAIN.RESUME = None
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCHS = 100
# early stopping
_C.TRAIN.EARLY_STOP = -1

# Tensorboard
_C.TRAIN.TENSORBOARD = CN()
_C.TRAIN.TENSORBOARD.ENABLE = True

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.BASE_LR = 1e-3
# FPN Learning rate
_C.TRAIN.OPTIMIZER.FP_LR = 4e-5
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# Weight decay
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = 'reduce'
# NoamLR parameters
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS = 2.0
_C.TRAIN.LR_SCHEDULER.INIT_LR = 1e-4
_C.TRAIN.LR_SCHEDULER.MAX_LR = 1e-2
_C.TRAIN.LR_SCHEDULER.FINAL_LR = 1e-4
# ReduceLRonPlateau
_C.TRAIN.LR_SCHEDULER.FACTOR = 0.7
_C.TRAIN.LR_SCHEDULER.PATIENCE = 10
_C.TRAIN.LR_SCHEDULER.MIN_LR = 1e-5


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(cfg, args):
    _update_config_from_file(cfg, args.cfg)

    cfg.defrost()
    if args.opts:
        cfg.merge_from_list(args.opts)
    # merge from specific arguments
    if args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    if args.lr_scheduler:
        cfg.TRAIN.LR_SCHEDULER.TYPE = args.lr_scheduler
    if args.resume:
        cfg.TRAIN.RESUME = args.resume
    if args.tag:
        cfg.TAG = args.tag
    if args.eval:
        cfg.EVAL_MODE = True

    # output folder
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.TAG)

    cfg.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    update_config(cfg, args)

    return cfg
