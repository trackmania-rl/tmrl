# standard library imports
import datetime
import os
import socket
import time
import atexit
import json
import shutil
import tempfile
from os.path import exists

# third-party imports
import numpy as np
from requests import get
from tlspyo import Relay, Endpoint

# local imports
from tmrl.actor import ActorModule
from tmrl.util import dump, load, partial_to_dict
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

import logging


__docformat__ = "google"


# PRINT: ============================================


def print_with_timestamp(s):
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    logging.info(sx + str(s))


def print_ip():
    public_ip = get('http://api.ipify.org').text
    local_ip = socket.gethostbyname(socket.gethostname())
    print_with_timestamp(f"public IP: {public_ip}, local IP: {local_ip}")


# BUFFER: ===========================================


class Buffer:
    """
    Buffer of training samples.

    `Server`, `RolloutWorker` and `Trainer` all have their own `Buffer` to store and send training samples.

    Samples are tuples of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
    """
    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        """
        Args:
            maxlen (int): buffer length
        """
        self.memory = []
        self.stat_train_return = 0.0  # stores the train return
        self.stat_test_return = 0.0  # stores the test return
        self.stat_train_steps = 0  # stores the number of steps per training episode
        self.stat_test_steps = 0  # stores the number of steps per test episode
        self.maxlen = maxlen

    def clip_to_maxlen(self):
        lenmem = len(self.memory)
        if lenmem > self.maxlen:
            print_with_timestamp("buffer overflow. Discarding old samples.")
            self.memory = self.memory[(lenmem - self.maxlen):]

    def append_sample(self, sample):
        """
        Appends `sample` to the buffer.

        Args:
            sample (Tuple): a training sample of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
        """
        self.memory.append(sample)
        self.clip_to_maxlen()

    def clear(self):
        """
        Clears the buffer but keeps train and test returns.
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        self.clip_to_maxlen()
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        self.stat_train_steps = other.stat_train_steps
        self.stat_test_steps = other.stat_test_steps
        return self


# SERVER SERVER: =====================================


class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """
    def __init__(self,
                 port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_port=cfg.LOCAL_PORT_SERVER,
                 header_size=cfg.HEADER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 max_workers=cfg.NB_WORKERS):
        """
        Args:
            port (int): tlspyo public port
            password (str): tlspyo password
            local_port (int): tlspyo local communication port
            header_size (int): tlspyo header size (bytes)
            security (Union[str, None]): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory
            max_workers (int): max number of accepted workers
        """
        self.__relay = Relay(port=port,
                             password=password,
                             accepted_groups={
                                 'trainers': {
                                     'max_count': 1,
                                     'max_consumables': None},
                                 'workers': {
                                     'max_count': max_workers,
                                     'max_consumables': None}},
                             local_com_port=local_port,
                             header_size=header_size,
                             security=security,
                             keys_dir=keys_dir)


# TRAINER: ==========================================


class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """
    def __init__(self,
                 server_ip=None,
                 server_port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_com_port=cfg.LOCAL_PORT_TRAINER,
                 header_size=cfg.HEADER_SIZE,
                 max_buf_len=cfg.BUFFER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 hostname=cfg.HOSTNAME,
                 model_path=cfg.MODEL_PATH_TRAINER):
        self.model_path = model_path
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.__endpoint = Endpoint(ip_server=self.server_ip,
                                   port=server_port,
                                   password=password,
                                   groups="trainers",
                                   local_com_port=local_com_port,
                                   header_size=header_size,
                                   max_buf_len=max_buf_len,
                                   security=security,
                                   keys_dir=keys_dir,
                                   hostname=hostname)

        print_with_timestamp(f"server IP: {self.server_ip}")

        self.__endpoint.notify(groups={'trainers': -1})  # retrieve everything

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
        model.save(self.model_path)
        with open(self.model_path, 'rb') as f:
            weights = f.read()
        self.__endpoint.broadcast(weights, "workers")

    def retrieve_buffer(self):
        """
        returns the TrainerInterface's buffer of training samples
        """
        buffers = self.__endpoint.receive_all()
        res = Buffer()
        for buf in buffers:
            res += buf
        self.__endpoint.notify(groups={'trainers': -1})  # retrieve everything
        return res


def log_environment_variables():
    """
    add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}


def load_run_instance(checkpoint_path):
    """
    Default function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    return load(checkpoint_path)


def dump_run_instance(run_instance, checkpoint_path):
    """
    Default function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    dump(run_instance, checkpoint_path)


def iterate_epochs_tm(run_cls,
                      interface: TrainerInterface,
                      checkpoint_path: str,
                      dump_run_instance_fn=dump_run_instance,
                      load_run_instance_fn=load_run_instance,
                      epochs_between_checkpoints=1,
                      updater_fn=None):
    """
    Main training loop (remote)
    The run_cls instance is saved in checkpoint_path at the end of each epoch
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        logging.debug(f"checkpoint_path: {checkpoint_path}")
        if not exists(checkpoint_path):
            logging.info(f"=== specification ".ljust(70, "="))
            run_instance = run_cls()
            dump_run_instance_fn(run_instance, checkpoint_path)
            logging.info(f"")
        else:
            logging.info(f"Loading checkpoint...")
            t1 = time.time()
            run_instance = load_run_instance_fn(checkpoint_path)
            logging.info(f" Loaded checkpoint in {time.time() - t1} seconds.")
            if updater_fn is not None:
                logging.info(f"Updating checkpoint...")
                t1 = time.time()
                run_instance = updater_fn(run_instance, run_cls)
                logging.info(f"Checkpoint updated in {time.time() - t1} seconds.")

        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch(interface=interface)  # yield stats data frame (this makes this function a generator)
            if run_instance.epoch % epochs_between_checkpoints == 0:
                logging.info(f" saving checkpoint...")
                t1 = time.time()
                dump_run_instance_fn(run_instance, checkpoint_path)
                logging.info(f" saved checkpoint in {time.time() - t1} seconds.")
                # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
                # del run_instance
                # gc.collect()  # garbage collection
                # run_instance = load_run_instance_fn(checkpoint_path)

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).

    saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    import wandb
    logging.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config['environ'] = log_environment_variables()
    # config['git'] = git_info()  # TODO: check this for bugs
    resume = checkpoint_path and exists(checkpoint_path)
    wandb_initialized = False
    err_cpt = 0
    while not wandb_initialized:
        try:
            wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
            wandb_initialized = True
        except Exception as e:
            err_cpt += 1
            logging.warning(f"wandb error {err_cpt}: {e}")
            if err_cpt > 10:
                logging.warning(f"Could not connect to wandb, aborting.")
                exit()
            else:
                time.sleep(10.0)
    # logging.info(config)
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run(interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        pass


class Trainer:
    """
    Training entity.

    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """
    def __init__(self,
                 training_cls=cfg_obj.TRAINER,
                 server_ip=cfg.SERVER_IP_FOR_TRAINER,
                 server_port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_com_port=cfg.LOCAL_PORT_TRAINER,
                 header_size=cfg.HEADER_SIZE,
                 max_buf_len=cfg.BUFFER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 hostname=cfg.HOSTNAME,
                 model_path=cfg.MODEL_PATH_TRAINER,
                 checkpoint_path=cfg.CHECKPOINT_PATH,
                 dump_run_instance_fn: callable = None,
                 load_run_instance_fn: callable = None,
                 updater_fn: callable = None):
        """
        Args:
            training_cls (type): training class (subclass of tmrl.training_offline.TrainingOffline)
            server_ip (str): ip of the central `Server`
            server_port (int): public port of the central `Server`
            password (str): password of the central `Server`
            local_com_port (int): port used by `tlspyo` for local communication
            header_size (int): number of bytes used for `tlspyo` headers
            max_buf_len (int): maximum number of messages queued by `tlspyo`
            security (str): `tlspyo security type` (None or "TLS")
            keys_dir (str): custom credentials directory for `tlspyo` TLS security
            hostname (str): custom TLS hostname
            model_path (str): path where a local copy of the model will be saved
            checkpoint_path: path where the `Trainer` will be checkpointed (`None` = no checkpointing)
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)
            updater_fn (callable): custom updater (`None` = no updater). If provided, this must be a function \
            that takes a checkpoint and training_cls as argument and returns an updated checkpoint. \
            The updater is called after a checkpoint is loaded, e.g., to update your checkpoint with new arguments.
        """
        self.checkpoint_path = checkpoint_path
        self.dump_run_instance_fn = dump_run_instance_fn
        self.load_run_instance_fn = load_run_instance_fn
        self.updater_fn = updater_fn
        self.training_cls = training_cls
        self.interface = TrainerInterface(server_ip=server_ip,
                                          server_port=server_port,
                                          password=password,
                                          local_com_port=local_com_port,
                                          header_size=header_size,
                                          max_buf_len=max_buf_len,
                                          security=security,
                                          keys_dir=keys_dir,
                                          hostname=hostname,
                                          model_path=model_path)

    def run(self):
        """
        Runs training.
        """
        run(interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn)

    def run_with_wandb(self,
                       entity=cfg.WANDB_ENTITY,
                       project=cfg.WANDB_PROJECT,
                       run_id=cfg.WANDB_RUN_ID,
                       key=None):
        """
        Runs training while logging metrics to wandb_.

        .. _wandb: https://wandb.ai

        Args:
            entity (str): wandb entity
            project (str): wandb project
            run_id (str): name of the run
            key (str): wandb API key
        """
        if key is not None:
            os.environ['WANDB_API_KEY'] = key
        run_with_wandb(entity=entity,
                       project=project,
                       run_id=run_id,
                       interface=self.interface,
                       run_cls=self.training_cls,
                       checkpoint_path=self.checkpoint_path,
                       dump_run_instance_fn=self.dump_run_instance_fn,
                       load_run_instance_fn=self.load_run_instance_fn,
                       updater_fn=self.updater_fn)


# ROLLOUT WORKER: ===================================


class RolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """
    def __init__(
            self,
            env_cls,
            actor_module_cls,
            sample_compressor: callable = None,
            device="cpu",
            max_samples_per_episode=np.inf,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,
            standalone=False,
            server_ip=None,
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            local_port=cfg.LOCAL_PORT_WORKER,
            header_size=cfg.HEADER_SIZE,
            max_buf_len=cfg.BUFFER_SIZE,
            security=cfg.SECURITY,
            keys_dir=cfg.CREDENTIALS_DIRECTORY,
            hostname=cfg.HOSTNAME
    ):
        """
        Args:
            env_cls (type): class of the Gymnasium environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): class of the module containing the policy (subclass of tmrl.actor.ActorModule)
            sample_compressor (callable): compressor for sending samples over the Internet; \
            when not `None`, `sample_compressor` must be a function that takes the following arguments: \
            (prev_act, obs, rew, terminated, truncated, info), and that returns them (modified) in the same order: \
            when not `None`, a `sample_compressor` works with a corresponding decompression scheme in the `Memory` class
            device (str): device on which the policy is running
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): utility for modifying observations retrieved from the environment; \
            when not `None`, `obs_preprocessor` must be a function that takes an observation as input and outputs the \
            modified observation
            crc_debug (bool): useful for debugging custom pipelines; leave to False otherwise
            model_path_history (str): (include the filename but omit .tmod) path to the saved history of policies; \
            we recommend you leave this to the default
            model_history (int): policies are saved every `model_history` new policies (0: not saved)
            standalone (bool): if True, the worker will not try to connect to a server
            server_ip (str): ip of the central server
            server_port (int): public port of the central server
            password (str): tlspyo password
            local_port (int): tlspyo local communication port; usually, leave this to the default
            header_size (int): tlspyo header size (bytes)
            max_buf_len (int): tlspyo max number of messages in buffer
            security (str): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory; usually, leave this to the default
            hostname (str): tlspyo hostname; usually, leave this to the default
        """
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.device = device
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to_device(self.device)
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logging.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logging.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0

        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'

        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            self.__endpoint = Endpoint(ip_server=self.server_ip,
                                       port=server_port,
                                       password=password,
                                       groups="workers",
                                       local_com_port=local_port,
                                       header_size=header_size,
                                       max_buf_len=max_buf_len,
                                       security=security,
                                       keys_dir=keys_dir,
                                       hostname=hostname,
                                       deserializer_mode="synchronous")
        else:
            self.__endpoint = None

    def act(self, obs, test=False):
        """
        Select an action based on observation `obs`

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the `act()` method of the `ActorModule`

        Returns:
            numpy.array: action computed by the `ActorModule`
        """
        # if self.obs_preprocessor is not None:
        #     obs = self.obs_preprocessor(obs)
        action = self.actor.act_(obs, test=test)
        return action

    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            Tuple:
            (nested structure: observation retrieved from the environment,
            dict: information retrieved from the environment)
        """
        obs = None
        act = self.env.default_action.astype(np.float32)
        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        terminated, truncated = False, False
        if collect_samples:
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is `obs` -> `act` -> `new_obs`, `rew`, `terminated`, `truncated`, `info`.
        Note that, in the Real-Time RL setting, `act` is appended to a buffer which is part of `new_obs`.
        This is because is does not directly affect the new observation, due to real-time delays.

        Args:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `terminated` is False, `truncated` will be set to True

        Returns:
            Tuple:
            (nested structure: new observation,
            float: new reward,
            bool: episode termination signal,
            bool: episode truncation signal,
            dict: information dictionary)
        """
        act = self.act(obs, test=test)
        new_obs, rew, terminated, truncated, info = self.env.step(act)
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
            if self.crc_debug:
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        return new_obs, rew, terminated, truncated, info

    def collect_train_episode(self, max_samples):
        """
        Collects a maximum of `max_samples` training transitions (from reset to terminated or truncated)

        This method stores the episode and the training return in the local `Buffer` of the worker
        for sending to the `Server`.

        Args:
            max_samples (int): if the environment is not `terminated` after `max_samples` time steps,
                it is forcefully reset and `truncated` is set to True.
        """
        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=True)
        for i in range(max_samples):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples_per_episode, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        counter = 0
        while counter < nb_episodes:
            self.run_episode(max_samples_per_episode, train=train)
            counter += 1

    def run_episode(self, max_samples, train=False):
        """
        Collects a maximum of `max_samples` test transitions (from reset to terminated or truncated).

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=False)
        for _ in range(max_samples):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=not train, collect_samples=False)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=50, nb_episodes=np.inf):  # TODO: check number of collected samples are collected before sending
        """
        Runs the worker for `nb_episodes` episodes.

        This method is for training.
        It collects a test episode each `test_episode_interval` episodes.
        For deployment, use the `run_episodes` method instead.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes
            nb_episodes (int): maximum number of train episodes to collect
        """
        episode = 0
        while episode < nb_episodes:
            if episode % test_episode_interval == 0 and not self.crc_debug:
                print_with_timestamp("running test episode")
                self.run_episode(self.max_samples_per_episode, train=False)
            print_with_timestamp("collecting train episode")
            self.collect_train_episode(self.max_samples_per_episode)
            print_with_timestamp("copying buffer for sending")
            self.send_and_clear_buffer()
            print_with_timestamp("checking for new weights")
            self.update_actor_weights()
            episode += 1
            # if self.crc_debug:
            #     break

    def run_env_benchmark(self, nb_steps, test=False):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        Furthermore, the `"benchmark"` option of the rtgym configuration dictionary must be set to `True`.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
        """
        obs, info = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=test, collect_samples=False)
            if terminated or truncated:
                break
        print_with_timestamp(f"Benchmark results:\n{self.env.benchmarks()}")

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self.__endpoint.produce(self.buffer, "trainers")
        self.buffer.clear()

    def update_actor_weights(self):
        """
        Updates the actor with new weights received from the `Server` when available.
        """
        weights_list = self.__endpoint.get_last()
        if len(weights_list) > 0:
            weights = weights_list[-1]
            with open(self.model_path, 'wb') as f:
                f.write(weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod", 'wb') as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    print_with_timestamp("model weights saved in history")
            self.actor = self.actor.load(self.model_path, device=self.device)
            print_with_timestamp("model weights have been updated")
