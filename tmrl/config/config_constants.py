# standard library imports
import os
from pathlib import Path
import logging
# HIGH-LEVEL PRAGMAS: ==========================================

PRAGMA_EDOUARD_YANN_CC = 2  # 2 if MISTlab RTX3080, 1 if ComputeCanada, 0 else
PRAGMA_SERVER_ON_EDOUARD_YANN = 0  # 1 is server on Edouard's PC, 0 if server on Yann's PC
RUN_NAME = "SACv1_SPINUP_4_LIDAR_pretrained_test_7"

BUFFERS_MAXLEN = 2000  # Maximum length of the local buffers for RolloutWorkers, Server and TrainerInterface
RW_MAX_SAMPLES_PER_EPISODE = 1000  # If this number of timesteps is reached, the RolloutWorker will reset the episode

PRAGMA_TM2020_TMNF = True  # True if TM2020, False if TMNF
PRAGMA_LIDAR = True  # True if Lidar, False if images
PRAGMA_RNN = False  # True to use an RNN, False to use an MLP

PRAGMA_CUDA_TRAINING = True  # True if CUDA, False if CPU (trainer)
PRAGMA_CUDA_INFERENCE = False  # True if CUDA, False if CPU (rollout worker)

PRAGMA_GAMEPAD = True  # True to use gamepad, False to use keyboard

CONFIG_COGNIFLY = False  # if True, will override config with Cognifly's config

PRAGMA_DCAC = False  # True for DCAC, False for SAC

LOCALHOST_WORKER = True  # set to True for RolloutWorkers on the same machine as the Server
LOCALHOST_TRAINER = True  # set to True for Trainers on the same machine as the Server
PUBLIC_IP_SERVER = "173.179.182.4" if PRAGMA_SERVER_ON_EDOUARD_YANN else "45.74.221.204"

SERVER_IP_FOR_WORKER = PUBLIC_IP_SERVER if not LOCALHOST_WORKER else "127.0.0.1"
SERVER_IP_FOR_TRAINER = PUBLIC_IP_SERVER if not LOCALHOST_TRAINER else "127.0.0.1"

# ENVIRONMENT: =======================================================

LIDAR_BLACK_THRESHOLD = [55, 55, 55]  # [88, 88, 88] for tiny road, [55, 55, 55] FOR BASIC ROAD
REWARD_END_OF_TRACK = 0  # bonus reward at the end of the track
CONSTANT_PENALTY = 0  # should be <= 0 : added to the reward at each time step
SLEEP_TIME_AT_RESET = 1.5  # 1.5 to start in a Markov state with the lidar, 0.0 for saving replays
ACT_BUF_LEN = 2
IMG_HIST_LEN = 4  # 4 without RNN, 1 with RNN

# DEBUGGING AND BENCHMARKING: ===================================

CRC_DEBUG = False  # Only for checking the consistency of the custom networking methods, set it to False otherwise. Caution: difficult to handle if reset transitions are collected.
CRC_DEBUG_SAMPLES = 100  # Number of samples collected in CRC_DEBUG mode
PROFILE_TRAINER = False  # Will profile each epoch in the Trainer when True
BENCHMARK = False  # The environment will be benchmarked when this is True
SYNCHRONIZE_CUDA = False  # Set to True for profiling, False otherwise

# FILE SYSTEM: =================================================

PATH_FILE = Path(__file__)  # TODO: this won't work with PyPI or normal install
logging.debug(f" PATH_FILE:{PATH_FILE}")
PATH_DATA = PATH_FILE.absolute().parent.parent / 'data'
logging.debug(f" PATH_DATA:{PATH_DATA}")

MODEL_HISTORY = 10  # 0 for not saving history, x for saving model history every x new model received by RolloutWorker

MODEL_PATH_WORKER = str(PATH_DATA / "weights" / (RUN_NAME + ".pth"))
MODEL_PATH_SAVE_HISTORY = str(PATH_DATA / "weights" / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(PATH_DATA / "weights" / (RUN_NAME + "_t.pth"))
CHECKPOINT_PATH = str(PATH_DATA / "checkpoint" / RUN_NAME)
DATASET_PATH = str(PATH_DATA / "dataset")
REWARD_PATH = str(PATH_DATA / "reward" / "reward.pkl")

if PRAGMA_EDOUARD_YANN_CC == 1:  # Override some of these for Compute Canada
    if PRAGMA_SERVER_ON_EDOUARD_YANN == 1:  # Edouard
        MODEL_PATH_TRAINER = r"/home/yannbout/scratch/base_tmrl_edouard/data/" + (RUN_NAME + "_t.pth")
        CHECKPOINT_PATH = r"/home/yannbout/scratch/base_tmrl_edouard/data/" + RUN_NAME
        REWARD_PATH = r"/home/yannbout/scratch/base_tmrl_edouard/data/reward.pkl"
    else:  # Yann
        MODEL_PATH_TRAINER = r"/home/yannbout/scratch/base_tmrl/data/" + (RUN_NAME + "_t.pth")
        CHECKPOINT_PATH = r"/home/yannbout/scratch/base_tmrl/data/" + RUN_NAME
        REWARD_PATH = r"/home/yannbout/scratch/base_tmrl/data/reward.pkl"
elif PRAGMA_EDOUARD_YANN_CC == 2:  # Override some of these for MIST Benchbot
    MODEL_PATH_TRAINER = r"/home/ybouteiller/tmrl/data/" + (RUN_NAME + "_t.pth")
    CHECKPOINT_PATH = r"/home/ybouteiller/tmrl/data/" + RUN_NAME
    REWARD_PATH = r"/home/ybouteiller/tmrl/data/reward.pkl"

# WANDB: =======================================================

WANDB_RUN_ID = RUN_NAME
WANDB_PROJECT = "tmrl"
WANDB_ENTITY = "tmrl"
WANDB_KEY = "df28d4daa98d2df2557d74caf78e40c68adaf288"

os.environ['WANDB_API_KEY'] = WANDB_KEY

# NETWORKING: ==================================================

PRINT_BYTESIZES = True

PORT_TRAINER = 55555  # Port to listen on (non-privileged ports are > 1023)
PORT_ROLLOUT = 55556  # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 536870912  # 268435456  # socket buffer size (200 000 000 is large enough for 1000 images right now)
HEADER_SIZE = 12  # fixed number of characters used to describe the data length

SOCKET_TIMEOUT_CONNECT_TRAINER = 300.0
SOCKET_TIMEOUT_ACCEPT_TRAINER = 300.0
SOCKET_TIMEOUT_CONNECT_ROLLOUT = 300.0
SOCKET_TIMEOUT_ACCEPT_ROLLOUT = 300.0  # socket waiting for rollout workers closed and restarted at this interval
SOCKET_TIMEOUT_COMMUNICATE = 30.0
SELECT_TIMEOUT_OUTBOUND = 30.0
SELECT_TIMEOUT_PING_PONG = 60.0
ACK_TIMEOUT_WORKER_TO_SERVER = 300.0
ACK_TIMEOUT_TRAINER_TO_SERVER = 300.0
ACK_TIMEOUT_SERVER_TO_WORKER = 300.0
ACK_TIMEOUT_SERVER_TO_TRAINER = 300.0
RECV_TIMEOUT_TRAINER_FROM_SERVER = 600.0
RECV_TIMEOUT_WORKER_FROM_SERVER = 600.0
WAIT_BEFORE_RECONNECTION = 10.0
LOOP_SLEEP_TIME = 1.0
