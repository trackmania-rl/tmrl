# third-party imports
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

# local imports
import tmrl.custom.config_constants as cfg
import tmrl.custom.config_objects as cfg_obj
from tmrl.custom.custom_gym_interfaces import TM2020InterfaceLidar
from tmrl.envs import UntouchedGymEnv
from tmrl.networking import RolloutWorker
from tmrl.util import partial


def run_car():
    config = cfg_obj.CONFIG_DICT
    config['interface_kwargs'] = {'save_replay': True}
    rw = RolloutWorker(env_cls=partial(UntouchedGymEnv, id="rtgym:real-time-gym-v0", gym_kwargs={"config": config}),
                       actor_module_cls=partial(cfg_obj.POLICY, act_buf_len=cfg.ACT_BUF_LEN),
                       get_local_buffer_sample=cfg_obj.SAMPLE_COMPRESSOR,
                       device='cuda' if cfg.PRAGMA_CUDA_INFERENCE else 'cpu',
                       redis_ip=cfg.REDIS_IP_FOR_WORKER,
                       samples_per_worker_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES,
                       model_path=cfg.MODEL_PATH_WORKER,
                       obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                       crc_debug=cfg.CRC_DEBUG)

    rw.run_episodes(10000)


if __name__ == "__main__":
    run_car()
