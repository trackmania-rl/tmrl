# third-party imports
import gymnasium
import cv2
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT
import os
from PIL import Image
from tmrl.networking import RolloutWorker
from tmrl.util import partial
from tmrl.envs import GenericGymEnv
import numpy as np

# local imports
from tmrl.custom.custom_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TM2020InterfaceLinux
from tmrl.custom.utils.window import WindowInterface
from tmrl.custom.utils.tools import Lidar
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
import logging


def check_env_tm20lidar():
    window_interface = WindowInterface("Trackmania")
    lidar = Lidar(window_interface.screenshot())
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceLidar
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False, "min_nb_steps_before_failure": int(20 * 60)}
    env = gymnasium.make("real-time-gym-v1", config=env_config)
    o, i = env.reset()
    while True:
        o, r, d, t, i = env.step(None)
        logging.info(f"r:{r}, d:{d}, t:{t}")
        if d or t:
            o, i = env.reset()
        img = window_interface.screenshot()[:, :, :3]
        lidar.lidar_20(img, True)


def show_imgs(imgs):
    imshape = imgs.shape
    if len(imshape) == 3:  # grayscale
        nb, h, w = imshape
        concat = imgs.reshape((nb*h, w))
        cv2.imshow("Environment", concat)
        cv2.waitKey(1)
    elif len(imshape) == 4:  # color
        nb, h, w, c = imshape
        concat = imgs.reshape((nb*h, w, c))
        cv2.imshow("Environment", concat)
        cv2.waitKey(1)

def save_obs(obs, counter, dir):
    speed = format(float(obs[0][0]), ".3f")
    gear = format(float(obs[1][0]), ".3f")
    rpm = format(float(obs[2][0]), ".3f")
    filename = f"{counter}-s_{speed}-g_{gear}-rpm_{rpm}.jpg"
    
    
    # revert preprocessing of images to save it
    image = obs[3][3] * 256.0
    image = image.astype(np.int8) 
    image = Image.fromarray(image).convert('RGB')
    image.save(os.path.join(dir, filename))


def check_env_tm20full():
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020Interface
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"gamepad": False,
                                      "min_nb_steps_before_failure": int(20 * 60),
                                      "grayscale": cfg.GRAYSCALE,
                                      "resize_to": (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)}
    env = gymnasium.make("real-time-gym-v1", config=env_config)
    o, i = env.reset()
    show_imgs(o[3])
    logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
    while True:
        o, r, d, t, i = env.step(None)
        show_imgs(o[3])
        logging.info(f"r:{r:.2f}, d:{d}, t:{t}, o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
        if d or t:
            o, i = env.reset()
            show_imgs(o[3])
            logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")

def check_headless_rw():
    config = cfg_obj.CONFIG_DICT
    rw = RolloutWorker(env_cls=partial(GenericGymEnv, id="real-time-gym-v1", gym_kwargs={"config": config}),
                           actor_module_cls=cfg_obj.POLICY,
                           sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                           device='cuda' if cfg.CUDA_INFERENCE else 'cpu',
                           server_ip=cfg.SERVER_IP_FOR_WORKER,
                           max_samples_per_episode=1000,
                           model_path=cfg.MODEL_PATH_WORKER,
                           obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                           crc_debug=cfg.CRC_DEBUG,
                           standalone=True)
    # setup storing
    obs_counter = 0
    directory = "check_env_output"
    if os.path.isdir(directory):
        dir_counter = 1
        while os.path.isdir(f"{directory}_{dir_counter}"):
            dir_counter += 1
        directory = f"{directory}_{dir_counter}"
        os.mkdir(directory)
        print(f"created {directory}")
    else:
        os.mkdir(directory)
        print(f"created {directory}")    

    # gather observations
    o, i = rw.reset(False)
    save_obs(o, obs_counter, directory)
    logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
    for step_num in range(100):
        o, r, d, t, i = rw.step(o, False, False)
        save_obs(o, step_num, directory)
        logging.info(f"r:{r:.2f}, d:{d}, t:{t}, o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
        if d or t:
            o, i = rw.reset(False)
            save_obs(o, step_num, directory)
            logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
    rw.reset(False)
    
if __name__ == "__main__":
    # check_env_tm20lidar()
    # check_env_tm20full()
    check_headless_rw()
