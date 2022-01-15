# third-party imports

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.util import partial


def run_car():
    config = cfg_obj.CONFIG_DICT
    config['interface_kwargs'] = {'save_replay': True}
    rw = RolloutWorker(env_cls=partial(GenericGymEnv, id="rtgym:real-time-gym-v0", gym_kwargs={"config": config}),
                       actor_module_cls=partial(cfg_obj.POLICY, act_buf_len=cfg.ACT_BUF_LEN),
                       get_local_buffer_sample=cfg_obj.SAMPLE_COMPRESSOR,
                       device='cuda' if cfg.PRAGMA_CUDA_INFERENCE else 'cpu',
                       server_ip=cfg.SERVER_IP_FOR_WORKER,
                       samples_per_worker_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES,
                       model_path=cfg.MODEL_PATH_WORKER,
                       obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                       crc_debug=cfg.CRC_DEBUG)

    rw.run_episodes(10000)


if __name__ == "__main__":
    run_car()
