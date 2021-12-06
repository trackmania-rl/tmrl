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

PRAGMA_CUDA_TRAINING = TMRL_CONFIG["CUDA_TRAINING"]  # True if CUDA, False if CPU (trainer)
PRAGMA_CUDA_INFERENCE = TMRL_CONFIG["CUDA_INFERENCE"]  # True if CUDA, False if CPU (rollout worker)

PRAGMA_GAMEPAD = TMRL_CONFIG["VIRTUAL_GAMEPAD"]  # True to use gamepad, False to use keyboard

PRAGMA_DCAC = False  # True for DCAC, False for SAC

LOCALHOST_WORKER = TMRL_CONFIG["LOCALHOST_WORKER"]  # set to True for RolloutWorkers on the same machine as the Server
LOCALHOST_TRAINER = TMRL_CONFIG["LOCALHOST_TRAINER"]  # set to True for Trainers on the same machine as the Server
PUBLIC_IP_SERVER = TMRL_CONFIG["PUBLIC_IP_SERVER"]

SERVER_IP_FOR_WORKER = PUBLIC_IP_SERVER if not LOCALHOST_WORKER else "127.0.0.1"
SERVER_IP_FOR_TRAINER = PUBLIC_IP_SERVER if not LOCALHOST_TRAINER else "127.0.0.1"

# ENVIRONMENT: =======================================================

ENV_CONFIG = TMRL_CONFIG["ENV"]
RTGYM_INTERFACE = ENV_CONFIG["RTGYM_INTERFACE"]
PRAGMA_TM2020_TMNF = RTGYM_INTERFACE.startswith("TM20")  # True if TM2020, False if TMNF
PRAGMA_LIDAR = RTGYM_INTERFACE.endswith("LIDAR")  # True if Lidar, False if images
LIDAR_BLACK_THRESHOLD = [55, 55, 55]  # [88, 88, 88] for tiny road, [55, 55, 55] FOR BASIC ROAD
REWARD_END_OF_TRACK = 0  # bonus reward at the end of the track
CONSTANT_PENALTY = 0  # should be <= 0 : added to the reward at each time step
SLEEP_TIME_AT_RESET = ENV_CONFIG["SLEEP_TIME_AT_RESET"]  # 1.5 to start in a Markov state with the lidar
IMG_HIST_LEN = ENV_CONFIG["IMG_HIST_LEN"]  # 4 without RNN, 1 with RNN
ACT_BUF_LEN = ENV_CONFIG["RTGYM_CONFIG"]["act_buf_len"]

# DEBUGGING AND BENCHMARKING: ===================================

CRC_DEBUG = False  # Only for checking the consistency of the custom networking methods, set it to False otherwise. Caution: difficult to handle if reset transitions are collected.
CRC_DEBUG_SAMPLES = 100  # Number of samples collected in CRC_DEBUG mode
PROFILE_TRAINER = False  # Will profile each epoch in the Trainer when True
SYNCHRONIZE_CUDA = False  # Set to True for profiling, False otherwise

# FILE SYSTEM: =================================================

PATH_DATA = TMRL_FOLDER
logging.debug(f" PATH_DATA:{PATH_DATA}")

MODEL_HISTORY = TMRL_CONFIG["SAVE_MODEL_EVERY"]  # 0 for not saving history, x for saving model history every x new model received by RolloutWorker

MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".pth"))
MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.pth"))
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / RUN_NAME)
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

PORT_TRAINER = TMRL_CONFIG["PORT_TRAINER"]  # Port to listen on (non-privileged ports are > 1023)
PORT_ROLLOUT = TMRL_CONFIG["PORT_ROLLOUT"]  # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = TMRL_CONFIG["BUFFER_SIZE"]  # 268435456  # socket buffer size (200 000 000 is large enough for 1000 images right now)
HEADER_SIZE = TMRL_CONFIG["HEADER_SIZE"]  # fixed number of characters used to describe the data length

SOCKET_TIMEOUT_CONNECT_TRAINER = TMRL_CONFIG["SOCKET_TIMEOUT_CONNECT_TRAINER"]
SOCKET_TIMEOUT_ACCEPT_TRAINER = TMRL_CONFIG["SOCKET_TIMEOUT_ACCEPT_TRAINER"]
SOCKET_TIMEOUT_CONNECT_ROLLOUT = TMRL_CONFIG["SOCKET_TIMEOUT_CONNECT_ROLLOUT"]
SOCKET_TIMEOUT_ACCEPT_ROLLOUT = TMRL_CONFIG["SOCKET_TIMEOUT_ACCEPT_ROLLOUT"]  # socket waiting for rollout workers closed and restarted at this interval
SOCKET_TIMEOUT_COMMUNICATE = TMRL_CONFIG["SOCKET_TIMEOUT_COMMUNICATE"]
SELECT_TIMEOUT_OUTBOUND = TMRL_CONFIG["SELECT_TIMEOUT_OUTBOUND"]
ACK_TIMEOUT_WORKER_TO_SERVER = TMRL_CONFIG["ACK_TIMEOUT_WORKER_TO_SERVER"]
ACK_TIMEOUT_TRAINER_TO_SERVER = TMRL_CONFIG["ACK_TIMEOUT_TRAINER_TO_SERVER"]
ACK_TIMEOUT_SERVER_TO_WORKER = TMRL_CONFIG["ACK_TIMEOUT_SERVER_TO_WORKER"]
ACK_TIMEOUT_SERVER_TO_TRAINER = TMRL_CONFIG["ACK_TIMEOUT_SERVER_TO_TRAINER"]
RECV_TIMEOUT_TRAINER_FROM_SERVER = TMRL_CONFIG["RECV_TIMEOUT_TRAINER_FROM_SERVER"]
RECV_TIMEOUT_WORKER_FROM_SERVER = TMRL_CONFIG["RECV_TIMEOUT_WORKER_FROM_SERVER"]
WAIT_BEFORE_RECONNECTION = TMRL_CONFIG["WAIT_BEFORE_RECONNECTION"]
LOOP_SLEEP_TIME = TMRL_CONFIG["LOOP_SLEEP_TIME"]
