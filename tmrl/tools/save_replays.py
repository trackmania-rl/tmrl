from argparse import ArgumentParser

# third-party imports
import numpy as np

# local imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.util import partial


def save_replays(nb_replays=np.inf):
    config = cfg_obj.CONFIG_DICT
    config['interface_kwargs'] = {'save_replay': True}
    rw = RolloutWorker(env_cls=partial(GenericGymEnv, id="real-time-gym-v0", gym_kwargs={"config": config}),
                       actor_module_cls=partial(cfg_obj.POLICY, act_buf_len=cfg.ACT_BUF_LEN),
                       sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                       device='cuda' if cfg.PRAGMA_CUDA_INFERENCE else 'cpu',
                       server_ip=cfg.SERVER_IP_FOR_WORKER,
                       min_samples_per_worker_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES,
                       model_path=cfg.MODEL_PATH_WORKER,
                       obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                       crc_debug=cfg.CRC_DEBUG,
                       standalone=True)

    rw.run_episodes(10000, nb_episodes=nb_replays)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--nb_replays', type=int, default=np.inf, help='number of replays to record')
    args = parser.parse_args()
    save_replays(args.nb_replays)
