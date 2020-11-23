import numpy as np
import os
from agents.util import partial

from agents.sac_models import Mlp, MlpPolicy
from agents.custom.custom_models import Tm_hybrid_1, TMPolicy
from agents.custom.custom_gym_interfaces import TM2020InterfaceLidar, TMInterfaceLidar, TM2020Interface, TMInterface, CogniflyInterfaceTask1
from agents.custom.custom_preprocessors import obs_preprocessor_tm_act_in_obs, obs_preprocessor_tm_lidar_act_in_obs, sample_preprocessor_tm_lidar_act_in_obs, obs_preprocessor_cognifly
from agents.custom.custom_memories import get_local_buffer_sample, MemoryTMNFLidar, MemoryTMNF, MemoryTM2020, get_local_buffer_sample_tm20_imgs, get_local_buffer_sample_cognifly, MemoryCognifly

# HIGH-LEVEL PRAGMAS: ==========================================

PRAGMA_EDOUARD_YANN_CC = 2  # 2 if ComputeCanada, 1 if Edouard, 0 if Yann  # TODO: remove for release
PRAGMA_TM2020_TMNF = True  # True if TM2020, False if TMNF
PRAGMA_LIDAR = True  # True if Lidar, False if images
PRAGMA_CUDA = True  # True if CUDA, False if CPU
CONFIG_COGNIFLY = False  # if True, will override config with Cognifly's config

# CRC DEBUGGING: ===============================================

CRC_DEBUG = False  # Only for checking the consistency of the custom networking methods, set it to False otherwise
CRC_DEBUG_SAMPLES = 10  # Number of samples collected in CRC_DEBUG mode

# BUFFERS: =====================================================

ACT_BUF_LEN = 2
IMG_HIST_LEN = 4

# FILE SYSTEM: =================================================

if PRAGMA_EDOUARD_YANN_CC == 2:  # Compute Canada
    # MODEL_PATH_TRAINER = r"/home/yannbout/scratch/base_tmrl/data/expt.pth"
    # CHECKPOINT_PATH = r"/home/yannbout/scratch/base_tmrl/data/exp0"
    MODEL_PATH_TRAINER = r"/home/yannbout/scratch/base_tmrl/data/expt1.pth"
    CHECKPOINT_PATH = r"/home/yannbout/scratch/base_tmrl/data/exp1"
    DATASET_PATH = r"/home/yannbout/scratch/base_tmrl/data/dataset"
    REWARD_PATH = r"/home/yannbout/scratch/base_tmrl/data/reward.pkl"
    MODEL_PATH_WORKER = r"/home/yannbout/scratch/base_tmrl/data/exp1.pth"
elif PRAGMA_EDOUARD_YANN_CC == 1:  # Edouard
    MODEL_PATH_WORKER = r"D:\cp\weights\exp1.pth"
    MODEL_PATH_TRAINER = r"D:\cp\weights\expt1.pth"
    CHECKPOINT_PATH = r"D:\cp\exp1"
    DATASET_PATH = r"D:\data2020"
    REWARD_PATH = r"D:\data2020reward\reward.pkl"
elif PRAGMA_EDOUARD_YANN_CC == 0:  # Yann
    MODEL_PATH_WORKER = r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\weights\exp1.pth"
    MODEL_PATH_TRAINER = r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\weights\expt1.pth"
    CHECKPOINT_PATH = r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\chk\exp1"
    DATASET_PATH = r"C:\Users\Yann\Desktop\git\tmrl\data"
    REWARD_PATH = r"C:/Users/Yann/Desktop/git/tmrl/tm20reward/reward.pkl"

# WANDB: =======================================================

WANDB_RUN_ID = "SAC_tm20_test_yann_03"
WANDB_PROJECT = "tmrl"
WANDB_ENTITY = "yannbouteiller"  # TODO: remove for release
WANDB_KEY = "9061c16ece78577b75f1a4af109a427d52b74b2a"  # TODO: remove for release

os.environ['WANDB_API_KEY'] = WANDB_KEY

# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

ACT_IN_OBS = (ACT_BUF_LEN > 0)
TRAIN_MODEL = Mlp if PRAGMA_LIDAR else Tm_hybrid_1
POLICY = MlpPolicy if PRAGMA_LIDAR else TMPolicy
BENCHMARK = False

if PRAGMA_LIDAR:
    INT = partial(TM2020InterfaceLidar, img_hist_len=IMG_HIST_LEN) if PRAGMA_TM2020_TMNF else partial(TMInterfaceLidar, img_hist_len=IMG_HIST_LEN)
else:
    INT = partial(TM2020Interface, img_hist_len=IMG_HIST_LEN) if PRAGMA_TM2020_TMNF else partial(TMInterface, img_hist_len=IMG_HIST_LEN)
CONFIG_DICT = {
    "interface": INT,
    "time_step_duration": 0.05,
    "start_obs_capture": 0.04,
    "time_step_timeout_factor": 1.0,
    "ep_max_length": np.inf,
    "real_time": True,
    "async_threading": True,
    "act_in_obs": ACT_IN_OBS,
    "act_buf_len": ACT_BUF_LEN,
    "benchmark": BENCHMARK,
    "wait_on_done": True,
}

# to compress a sample before sending it over the local network/Internet:
SAMPLE_COMPRESSOR = get_local_buffer_sample if PRAGMA_LIDAR else get_local_buffer_sample_tm20_imgs
# to preprocess observations that come out of the gym environment and of the replay buffer:
OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs if PRAGMA_LIDAR else obs_preprocessor_tm_act_in_obs
# to augment data that comes out of the replay buffer (applied after observation preprocessing):
SAMPLE_PREPROCESSOR = sample_preprocessor_tm_lidar_act_in_obs if PRAGMA_LIDAR else None

if PRAGMA_LIDAR:
    MEM = MemoryTMNFLidar
else:
    MEM = MemoryTM2020 if PRAGMA_TM2020_TMNF else MemoryTMNF
MEMORY = partial(MEM,
                 path_loc=DATASET_PATH,
                 imgs_obs=IMG_HIST_LEN,
                 act_buf_len=ACT_BUF_LEN,
                 obs_preprocessor=OBS_PREPROCESSOR,
                 sample_preprocessor=SAMPLE_PREPROCESSOR,
                 crc_debug=CRC_DEBUG
                 )

# NETWORKING: ==================================================

LOCALHOST = False  # set to True to enforce localhost
REDIS_IP = "96.127.215.210" if not LOCALHOST else "127.0.0.1"

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
ACK_TIMEOUT_WORKER_TO_REDIS = 300.0
ACK_TIMEOUT_TRAINER_TO_REDIS = 300.0
ACK_TIMEOUT_REDIS_TO_WORKER = 300.0
ACK_TIMEOUT_REDIS_TO_TRAINER = 300.0
WAIT_BEFORE_RECONNECTION = 10.0
LOOP_SLEEP_TIME = 1.0

# COGNIFLY: ====================================================

if CONFIG_COGNIFLY:

    if PRAGMA_EDOUARD_YANN_CC == 0:  # Yann  # TODO: CC
        MODEL_PATH_WORKER = r"/home/yann/Desktop/git/projets_perso/tmrl_cognifly_data/expcgn.pth"
        MODEL_PATH_TRAINER = r"/home/yann/Desktop/git/projets_perso/tmrl_cognifly_data/expcgnt.pth"
        CHECKPOINT_PATH = r"/home/yann/Desktop/git/projets_perso/tmrl_cognifly_data/expcgn0"

    WANDB_RUN_ID = "SAC_cognifly_test_2"
    WANDB_PROJECT = "cognifly"

    TRAIN_MODEL = Mlp
    POLICY = MlpPolicy
    BENCHMARK = False

    ACT_BUF_LEN = 4
    IMGS_OBS = 0

    INT = partial(CogniflyInterfaceTask1, img_hist_len=0)

    from rtgym import DEFAULT_CONFIG_DICT
    CONFIG_DICT = DEFAULT_CONFIG_DICT
    CONFIG_DICT["interface"] = INT

    CONFIG_DICT["time_step_duration"] = 0.05
    CONFIG_DICT["start_obs_capture"] = 0.05
    CONFIG_DICT["time_step_timeout_factor"] = 1.0
    CONFIG_DICT["ep_max_length"] = 200
    CONFIG_DICT["act_buf_len"] = ACT_BUF_LEN
    CONFIG_DICT["reset_act_buf"] = False
    CONFIG_DICT["act_in_obs"] = True
    CONFIG_DICT["benchmark"] = BENCHMARK
    CONFIG_DICT["wait_on_done"] = True

    SAMPLE_COMPRESSOR = get_local_buffer_sample_cognifly
    OBS_PREPROCESSOR = obs_preprocessor_cognifly
    SAMPLE_PREPROCESSOR = None

    MEM = MemoryCognifly

    MEMORY = partial(MEM,
                     path_loc=DATASET_PATH,
                     imgs_obs=IMGS_OBS,
                     act_buf_len=ACT_BUF_LEN,
                     obs_preprocessor=OBS_PREPROCESSOR,
                     sample_preprocessor=SAMPLE_PREPROCESSOR,
                     crc_debug=CRC_DEBUG
                     )
