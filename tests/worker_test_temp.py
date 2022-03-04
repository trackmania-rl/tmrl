# standard library imports
import socket
import sys
import time
from pathlib import Path
from threading import Lock, Thread

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.util import partial
import subprocess

script_file = Path(__file__).absolute().parent


class TM2020OpenPlanetServer:
    def __init__(self, host='127.0.0.1', port=9000):
        self._host = host
        self._port = port

        self.__t_client = Thread(target=self.__client_thread, args=(), kwargs={}, daemon=True)
        self.__t_client.start()

    def __client_thread(self):
        """
        Thread of the client.
        This listens for incoming data until the object is destroyed
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self._host, self._port))
            s.listen()
            conn, addr = s.accept()
            with conn:
                with open(script_file / "openplanet_record.txt", "rb") as f:
                    lines = f.readlines()
                    for line in lines:
                        conn.sendall(line)
                        time.sleep(0.05)


fakeServer = TM2020OpenPlanetServer()



subprocess.call(r"run.py --server", shell=True)
subprocess.call(r"run.py --trainer", shell=True)

worker = RolloutWorker(env_cls=partial(GenericGymEnv, id="real-time-gym-v0", gym_kwargs={"config": cfg_obj.CONFIG_DICT}),
                       actor_module_cls=partial(cfg_obj.POLICY, act_buf_len=cfg.ACT_BUF_LEN),
                       sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                       device='cuda' if cfg.PRAGMA_CUDA_INFERENCE else 'cpu',
                       server_ip=cfg.SERVER_IP_FOR_WORKER,
                       min_samples_per_worker_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES,
                       max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                       model_path=cfg.MODEL_PATH_WORKER,
                       obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                       crc_debug=cfg.CRC_DEBUG,
                       standalone=True)

worker.run()
