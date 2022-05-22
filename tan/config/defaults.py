import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "TAN"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256
_C.INPUT.PRE_QUERY_SIZE = 300

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.NAME = ""
_C.DATASETS.METRIC_TENSORBOARD = True

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.TAN = CN()
_C.MODEL.TAN.NUM_CLIPS = 128
_C.MODEL.TAN.JOINT_SPACE_SIZE = 128

_C.MODEL.TAN.FEATPOOL = CN()
_C.MODEL.TAN.FEATPOOL.INPUT_SIZE = [1024]
_C.MODEL.TAN.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.TAN.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.TAN.FEAT2D = CN()
_C.MODEL.TAN.FEAT2D.NAME = "pool"
_C.MODEL.TAN.FEAT2D.POOLING_COUNTS = [15, 8, 8, 8]

_C.MODEL.TAN.TEXT_ENCODER = CN()
_C.MODEL.TAN.TEXT_ENCODER.QUERY_HIDDEN_SIZE = 512
_C.MODEL.TAN.TEXT_ENCODER.NAME = 'BERT'
_C.MODEL.TAN.TEXT_ENCODER.LSTM = CN()
_C.MODEL.TAN.TEXT_ENCODER.LSTM.NUM_LAYERS = 3
_C.MODEL.TAN.TEXT_ENCODER.LSTM.BIDIRECTIONAL = False


_C.MODEL.TAN.PREDICTOR = CN() 
_C.MODEL.TAN.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.TAN.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.TAN.PREDICTOR.NUM_STACK_LAYERS = 8


_C.MODEL.TAN.LOSS = CN()
_C.MODEL.TAN.LOSS.MIN_IOU = 0.3
_C.MODEL.TAN.LOSS.MAX_IOU = 0.7
_C.MODEL.TAN.LOSS.BCE_WEIGHT = 1
_C.MODEL.TAN.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL = 5
_C.MODEL.TAN.LOSS.NEGATIVE_VIDEO_IOU = 0.5
_C.MODEL.TAN.LOSS.SENT_REMOVAL_IOU = 0.5
_C.MODEL.TAN.LOSS.PAIRWISE_SENT_WEIGHT = 0.0
_C.MODEL.TAN.LOSS.CONTRASTIVE_WEIGHT = 0.1
_C.MODEL.TAN.LOSS.TAU_VIDEO = 0.2
_C.MODEL.TAN.LOSS.TAU_SENT = 0.2
_C.MODEL.TAN.LOSS.MARGIN = 0.2

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_EPOCH = 1
_C.SOLVER.FREEZE_BERT = 4
_C.SOLVER.ONLY_IOU = 7
_C.SOLVER.SKIP_TEST = 0

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.5
_C.TEST.CONTRASTIVE_SCORE_POW = 0.5
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
