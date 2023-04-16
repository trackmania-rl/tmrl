# standard library imports
import os
from pathlib import Path
import logging
import json


TMRL_FOLDER = Path.home() / "TmrlData"
CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
DATASET_FOLDER = TMRL_FOLDER / "dataset"
REWARD_FOLDER = TMRL_FOLDER / "reward"
WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
CONFIG_FOLDER = TMRL_FOLDER / "config"

CONFIG_FILE = TMRL_FOLDER / "config" / "config.json"
with open(CONFIG_FILE) as f:
    TMRL_CONFIG = json.load(f)

RUN_NAME = TMRL_CONFIG["RUN_NAME"]  # "SACv1_SPINUP_4_LIDAR_pretrained_test_9"
BUFFERS_MAXLEN = TMRL_CONFIG["BUFFERS_MAXLEN"]  # Maximum length of the local buffers for RolloutWorkers, Server and TrainerInterface
RW_MAX_SAMPLES_PER_EPISODE = TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]  # If this number of timesteps is reached, the RolloutWorker will reset the episode

PRAGMA_RNN = False  # True to use an RNN, False to use an MLP

CUDA_TRAINING = TMRL_CONFIG["CUDA_TRAINING"]  # True if CUDA, False if CPU (trainer)
CUDA_INFERENCE = TMRL_CONFIG["CUDA_INFERENCE"]  # True if CUDA, False if CPU (rollout worker)

PRAGMA_GAMEPAD = TMRL_CONFIG["VIRTUAL_GAMEPAD"]  # True to use gamepad, False to use keyboard

LOCALHOST_WORKER = TMRL_CONFIG["LOCALHOST_WORKER"]  # set to True for RolloutWorkers on the same machine as the Server
LOCALHOST_TRAINER = TMRL_CONFIG["LOCALHOST_TRAINER"]  # set to True for Trainers on the same machine as the Server
PUBLIC_IP_SERVER = TMRL_CONFIG["PUBLIC_IP_SERVER"]

SERVER_IP_FOR_WORKER = PUBLIC_IP_SERVER if not LOCALHOST_WORKER else "127.0.0.1"
SERVER_IP_FOR_TRAINER = PUBLIC_IP_SERVER if not LOCALHOST_TRAINER else "127.0.0.1"

# ENVIRONMENT: =======================================================

ENV_CONFIG = TMRL_CONFIG["ENV"]
RTGYM_INTERFACE = ENV_CONFIG["RTGYM_INTERFACE"]
PRAGMA_LIDAR = RTGYM_INTERFACE.endswith("LIDAR")  # True if Lidar, False if images
PRAGMA_PROGRESS = RTGYM_INTERFACE.endswith("LIDARPROGRESS")
if PRAGMA_PROGRESS:
    PRAGMA_LIDAR = True
LIDAR_BLACK_THRESHOLD = [55, 55, 55]  # [88, 88, 88] for tiny road, [55, 55, 55] FOR BASIC ROAD
REWARD_END_OF_TRACK = 100  # bonus reward at the end of the track
CONSTANT_PENALTY = 0  # should be <= 0 : added to the reward at each time step
SLEEP_TIME_AT_RESET = ENV_CONFIG["SLEEP_TIME_AT_RESET"]  # 1.5 to start in a Markov state with the lidar
IMG_HIST_LEN = ENV_CONFIG["IMG_HIST_LEN"]  # 4 without RNN, 1 with RNN
ACT_BUF_LEN = ENV_CONFIG["RTGYM_CONFIG"]["act_buf_len"]
WINDOW_WIDTH = ENV_CONFIG["WINDOW_WIDTH"]
WINDOW_HEIGHT = ENV_CONFIG["WINDOW_HEIGHT"]
GRAYSCALE = ENV_CONFIG["IMG_GRAYSCALE"] if "IMG_GRAYSCALE" in ENV_CONFIG else False
IMG_WIDTH = ENV_CONFIG["IMG_WIDTH"] if "IMG_WIDTH" in ENV_CONFIG else 64
IMG_HEIGHT = ENV_CONFIG["IMG_HEIGHT"] if "IMG_HEIGHT" in ENV_CONFIG else 64

# DEBUGGING AND BENCHMARKING: ===================================

CRC_DEBUG = False  # Only for checking the consistency of the custom networking methods, set it to False otherwise. Caution: difficult to handle if reset transitions are collected.
CRC_DEBUG_SAMPLES = 100  # Number of samples collected in CRC_DEBUG mode
PROFILE_TRAINER = False  # Will profile each epoch in the Trainer when True
SYNCHRONIZE_CUDA = False  # Set to True for profiling, False otherwise
DEBUG_MODE = TMRL_CONFIG["DEBUG_MODE"] if "DEBUG_MODE" in TMRL_CONFIG.keys() else False

# FILE SYSTEM: =================================================

PATH_DATA = TMRL_FOLDER
logging.debug(f" PATH_DATA:{PATH_DATA}")

MODEL_HISTORY = TMRL_CONFIG["SAVE_MODEL_EVERY"]  # 0 for not saving history, x for saving model history every x new model received by RolloutWorker

MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".tmod"))
MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.tmod"))
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_t.tcpt"))
DATASET_PATH = str(DATASET_FOLDER)
REWARD_PATH = str(REWARD_FOLDER / "reward.pkl")

# WANDB: =======================================================

WANDB_RUN_ID = RUN_NAME
WANDB_PROJECT = TMRL_CONFIG["WANDB_PROJECT"]
WANDB_ENTITY = TMRL_CONFIG["WANDB_ENTITY"]
WANDB_KEY = TMRL_CONFIG["WANDB_KEY"]

os.environ['WANDB_API_KEY'] = WANDB_KEY

# NETWORKING: ==================================================

PRINT_BYTESIZES = True

PORT = TMRL_CONFIG["PORT"]  # Port to listen to (non-privileged ports are > 1023)
LOCAL_PORT_SERVER = TMRL_CONFIG["LOCAL_PORT_SERVER"]
LOCAL_PORT_TRAINER = TMRL_CONFIG["LOCAL_PORT_TRAINER"]
LOCAL_PORT_WORKER = TMRL_CONFIG["LOCAL_PORT_WORKER"]
PASSWORD = TMRL_CONFIG["PASSWORD"]
SECURITY = "TLS" if TMRL_CONFIG["TLS"] else None
CREDENTIALS_DIRECTORY = TMRL_CONFIG["TLS_CREDENTIALS_DIRECTORY"] if TMRL_CONFIG["TLS_CREDENTIALS_DIRECTORY"] != "" else None
HOSTNAME = TMRL_CONFIG["TLS_HOSTNAME"]
NB_WORKERS = None if TMRL_CONFIG["NB_WORKERS"] < 0 else TMRL_CONFIG["NB_WORKERS"]

BUFFER_SIZE = TMRL_CONFIG["BUFFER_SIZE"]  # 268435456  # socket buffer size (200 000 000 is large enough for 1000 images right now)
HEADER_SIZE = TMRL_CONFIG["HEADER_SIZE"]  # fixed number of characters used to describe the data length
