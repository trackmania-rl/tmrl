from agents.sac_models import Mlp, Tm_hybrid_1, TMPolicy, MlpPolicy
from agents.memory_dataloading import MemoryTMNFLidar, MemoryTM2020, MemoryTMNF
import numpy as np
from agents.util import  partial
from requests import get
from gym_tmrl.envs.tmrl_env import TM2020InterfaceLidar, TMInterfaceLidar, TM2020Interface, TMInterface
import socket
import os
# OBSERVATION PREPROCESSING ==================================

def obs_preprocessor_tm_act_in_obs(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    # print(f"DEBUG1: len(obs):{len(obs)}, obs[0]:{obs[0]}, obs[1].shape:{obs[1].shape}, obs[2]:{obs[2]}")
    # obs = (obs[0] / 1000.0, np.moveaxis(obs[1], -1, 0) / 255.0, obs[2])
    obs = (obs[0], np.moveaxis(obs[1], -1, 1), obs[2])
    # print(f"DEBUG2: len(obs):{len(obs)}, obs[0]:{obs[0]}, obs[1].shape:{obs[1].shape}, obs[2]:{obs[2]}")
    # exit()
    return obs


def obs_preprocessor_tm_lidar_act_in_obs(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    # print(f"DEBUG: obs:{obs}")
    # exit()
    obs = (obs[0], np.ndarray.flatten(obs[1]), obs[2])
    return obs


# WANDB: ==================================================

WANDB_RUN_ID = "tm2020_test_1"
WANDB_PROJECT = "tmrl"
WANDB_ENTITY = "yannbouteiller"
WANDB_KEY = "9061c16ece78577b75f1a4af109a427d52b74b2a"

os.environ['WANDB_API_KEY'] = WANDB_KEY


# CONFIGURATION: ==========================================

PRAGMA_EDOUARD_YANN = False  # True if Edouard, False if Yann
PRAGMA_TM2020_TMNF = False  # True if TM2020, False if TMNF
PRAGMA_LIDAR = True  # True if Lidar, False if images
PRAGMA_CUDA = False  # True if CUDA, False if CPU

TRAIN_MODEL = Mlp if PRAGMA_LIDAR else Tm_hybrid_1
POLICY = MlpPolicy if PRAGMA_LIDAR else TMPolicy

OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs if PRAGMA_LIDAR else obs_preprocessor_tm_act_in_obs

ACT_IN_OBS = True
BENCHMARK = False

public_ip = get('http://api.ipify.org').text
local_ip = socket.gethostbyname(socket.gethostname())
print(f"I: local IP: {local_ip}")
print(f"I: public IP: {public_ip}")
REDIS_IP = "96.127.215.210"  # public_ip
LOCALHOST = False

PORT_TRAINER = 55555  # Port to listen on (non-privileged ports are > 1023)
PORT_ROLLOUT = 55556  # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 268435456  # 1048576  # 8192  # 32768  # socket buffer size (200 000 000 is large enough for 1000 images right now)
HEADER_SIZE = 12  # fixed number of characters used to describe the data length

SOCKET_TIMEOUT_CONNECT_TRAINER = 60.0
SOCKET_TIMEOUT_ACCEPT_TRAINER = 60.0
SOCKET_TIMEOUT_CONNECT_ROLLOUT = 60.0
SOCKET_TIMEOUT_ACCEPT_ROLLOUT = 60.0  # socket waiting for rollout workers closed and restarted at this interval
SOCKET_TIMEOUT_COMMUNICATE = 30.0
SELECT_TIMEOUT_OUTBOUND = 30.0
SELECT_TIMEOUT_PING_PONG = 60.0
ACK_TIMEOUT_WORKER_TO_REDIS = 60.0
ACK_TIMEOUT_TRAINER_TO_REDIS = 60.0
ACK_TIMEOUT_REDIS_TO_WORKER = 60.0
ACK_TIMEOUT_REDIS_TO_TRAINER = 60.0
WAIT_BEFORE_RECONNECTION = 10.0
LOOP_SLEEP_TIME = 1.0

MODEL_PATH_WORKER = r"D:\cp\weights\exp.pth" if PRAGMA_EDOUARD_YANN else r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\weights\exp.pth"
MODEL_PATH_TRAINER = r"D:\cp\weights\expt.pth" if PRAGMA_EDOUARD_YANN else r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\weights\expt.pth"
CHECKPOINT_PATH = r"D:\cp\exp0" if PRAGMA_EDOUARD_YANN else r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\chk\exp0"
DATASET_PATH = r"D:\data2020" if PRAGMA_EDOUARD_YANN else r"C:\Users\Yann\Desktop\git\tmrl\data"

if PRAGMA_LIDAR:
    MEM = MemoryTMNFLidar
else:
    MEM = MemoryTM2020 if PRAGMA_TM2020_TMNF else MemoryTMNF

MEMORY = partial(MEM,
                 path_loc=DATASET_PATH,
                 imgs_obs=1 if PRAGMA_LIDAR else 4,
                 act_in_obs=ACT_IN_OBS,
                 obs_preprocessor=OBS_PREPROCESSOR
                 )

if PRAGMA_LIDAR:
    INT = partial(TM2020InterfaceLidar, img_hist_len=1) if PRAGMA_TM2020_TMNF else partial(TMInterfaceLidar, img_hist_len=1)
else:
    INT = TM2020Interface if PRAGMA_TM2020_TMNF else TMInterface

CONFIG_DICT = {
    "interface": INT,
    "time_step_duration": 0.05,
    "start_obs_capture": 0.04,
    "time_step_timeout_factor": 1.0,
    "ep_max_length": np.inf,
    "real_time": True,
    "async_threading": True,
    "act_in_obs": ACT_IN_OBS,
    "benchmark": BENCHMARK,
}
