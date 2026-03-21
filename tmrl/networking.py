"""Server, buffer, trainer, and rollout worker for distributed TMRL training."""

import atexit
import contextlib
import datetime
import itertools
import math
import os
import pickle
import shutil
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Generator
from os.path import exists
from typing import Any

import gymnasium
import numpy as np
from loguru import logger
from requests import get  # type: ignore[import-untyped]
from tlspyo import Endpoint, Relay
from tlspyo.server import Server as TlspyoServer

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
import tmrl.config.constants as cfg_c
from tmrl.actor import ActorModule
from tmrl.util import dump, load, partial_to_dict

__docformat__ = "google"


def print_with_timestamp(message: str) -> None:
    """Log message with current date/time prefix."""
    timestamp = datetime.datetime.now().strftime("%x %X ")
    logger.info("{}{}", timestamp, message)


def print_ip():
    try:
        public_ip = get("http://api.ipify.org", timeout=5).text
    except Exception:
        public_ip = "unavailable"
    local_ip = socket.gethostbyname(socket.gethostname())
    print_with_timestamp(f"public IP: {public_ip}, local IP: {local_ip}")


def _start_relay_windows_tcp(
    port: int,
    password: str,
    local_port: int,
    header_size: int,
    max_workers: int,
):
    """Run tlspyo relay server in a thread on Windows when TLS is disabled.

    The default tlspyo Relay uses a subprocess for the server; on Windows the child's
    stderr is often not visible, so bind failures (e.g. port in use) are silent. This
    runs the same server logic in a thread so any exception is visible in the same process.
    """
    local_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    local_srv.bind(("127.0.0.1", local_port))
    local_srv.listen()

    accepted_groups = {
        "trainers": {"max_count": 1, "max_consumables": None},
        "workers": {"max_count": max_workers, "max_consumables": None},
    }
    server = TlspyoServer(
        port=port,
        password=password,
        serializer=pickle.dumps,
        deserializer=pickle.loads,
        accepted_groups=accepted_groups,
        local_com_port=local_port,
        header_size=header_size,
        security="TCP",
        keys_dir=None,
    )

    def run_server():
        from twisted.internet import reactor

        _orig_run = reactor.run

        def run_without_signals(install_signal_handlers=0):
            return _orig_run(installSignalHandlers=install_signal_handlers)

        reactor.run = run_without_signals
        try:
            server.run()
        except Exception as e:
            logger.exception(
                "Relay server thread failed (this is the process that should bind to port {}): {}",
                port,
                e,
            )
            raise

    thread = threading.Thread(target=run_server, daemon=False)
    thread.start()

    conn, _ = local_srv.accept()
    msg = server.serializer(("TEST", None))
    header = bytes(f"{len(msg):<{header_size}}", "utf-8")
    conn.sendall(header + msg)

    return type("_WindowsTcpRelay", (), {"_thread": thread, "_conn": conn, "_srv": local_srv})()


class Buffer:
    """In-memory buffer of transition samples for the Server, RolloutWorker, and Trainer.

    Samples are tuples: (act, new_obs, rew, terminated, truncated, info).

    Intended for single-threaded use (one Buffer per worker or per trainer side). If the same
    Buffer instance is ever shared across threads, construct with thread_safe=True so that
    append_sample, clip_to_maxlen, and __iadd__ are guarded by a lock.
    """

    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN, thread_safe: bool = False):
        """Initialize an empty buffer with optional max length.

        Args:
            maxlen: Maximum number of samples to keep; older samples are dropped when exceeded.
            thread_safe: If True, use a lock around memory updates (for future multi-threaded use).
        """
        self.memory: list[Any] = []
        self.stat_train_return = 0.0
        self.stat_test_return = 0.0
        self.stat_train_steps = 0
        self.stat_test_steps = 0
        self.stat_test_finish_time = 0.0
        self.stat_test_finished_track = False
        self.stat_test_finished_count = 0
        self.stat_test_competition_eliminated = False
        self.stat_test_competition_crashes = 0
        self.maxlen = maxlen
        self._lock = threading.RLock() if thread_safe else None

    @contextlib.contextmanager
    def _guarded(self) -> Generator[None, None, None]:
        """Acquire the buffer lock if thread_safe, else no-op. Uses RLock so nesting is safe."""
        if self._lock is not None:
            self._lock.acquire()
        try:
            yield
        finally:
            if self._lock is not None:
                self._lock.release()

    def clip_to_maxlen(self):
        with self._guarded():
            lenmem = len(self.memory)
            if lenmem > self.maxlen:
                print_with_timestamp("buffer overflow. Discarding old samples.")
                self.memory = self.memory[(lenmem - self.maxlen) :]

    def append_sample(self, sample):
        """Append a sample ``(act, new_obs, rew, terminated, truncated, info)`` to the buffer."""
        with self._guarded():
            self.memory.append(sample)
            self.clip_to_maxlen()

    def clear(self):
        """Clear the buffer but keep train and test return stats."""
        with self._guarded():
            self.memory = []

    def apply_speed_bonus(self, speed_scale: float) -> None:
        """Spread a time/speed bonus over all rewards in this episode (in-place), K/T² formula.

        Each step gets speed_scale / T² so the total bonus is speed_scale / T;
        faster episodes (smaller T) get a higher total and every step carries the signal.
        Avoids a terminal spike that harms TQC convergence and long-horizon credit assignment.

        Call this after collecting a full episode, before sending the buffer.
        No-op if speed_scale <= 0 or buffer is empty.

        Args:
            speed_scale: Typically TIME_BONUS_SCALE * REWARD_SCALE
                (rewards in buffer are already scaled).
        """
        if speed_scale <= 0 or len(self.memory) == 0:
            return
        with self._guarded():
            num_steps = len(self.memory)
            bonus_per_step = speed_scale / (num_steps * num_steps)
            total_bonus = bonus_per_step * num_steps
            new_memory = []
            old_total = 0.0
            for _i, sample in enumerate(self.memory):
                act, obs, rew, term, trunc, info = sample
                old_total += rew
                new_rew = rew + bonus_per_step
                new_info = dict(info) if isinstance(info, dict) else info
                new_memory.append((act, obs, new_rew, term, trunc, new_info))
            new_total = old_total + total_bonus
            if new_memory and isinstance(new_memory[-1][5], dict):
                new_memory[-1][5]["reward_sum"] = new_total
            self.memory = new_memory
            self.stat_train_return = new_total

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        with self._guarded():
            self.memory += other.memory
            self.clip_to_maxlen()
            self.stat_train_return = other.stat_train_return
            self.stat_test_return = other.stat_test_return
            self.stat_train_steps = other.stat_train_steps
            self.stat_test_steps = other.stat_test_steps
            self.stat_test_finish_time = getattr(other, "stat_test_finish_time", 0.0)
            self.stat_test_finished_track = getattr(other, "stat_test_finished_track", False)
            self.stat_test_finished_count = getattr(other, "stat_test_finished_count", 0)
            self.stat_test_competition_eliminated = getattr(
                other, "stat_test_competition_eliminated", False
            )
            self.stat_test_competition_crashes = getattr(other, "stat_test_competition_crashes", 0)
            return self


class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """

    def __init__(
        self,
        port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_SERVER,
        header_size=cfg.HEADER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        max_workers=cfg.NB_WORKERS,
    ):
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
        if sys.platform == "win32" and security is None:
            self.__relay = _start_relay_windows_tcp(
                port=port,
                password=password,
                local_port=local_port,
                header_size=header_size,
                max_workers=max_workers,
            )
        else:
            self.__relay = Relay(
                port=port,
                password=password,
                accepted_groups={
                    "trainers": {"max_count": 1, "max_consumables": None},
                    "workers": {"max_count": max_workers, "max_consumables": None},
                },
                local_com_port=local_port,
                header_size=header_size,
                security=security,
                keys_dir=keys_dir,
            )
        import logging

        logging.getLogger("tlspyo").setLevel(logging.INFO)
        print_with_timestamp(
            f"TMRL server listening on port {port} (trainers + workers). "
            "Leave this process running."
        )
        try:
            config_path = str(cfg.CONFIG_FILE_PATH)
        except AttributeError:
            config_path = "(config path not available)"
        print_with_timestamp(
            f"Config: {config_path} (ensure server, trainer, worker use this same config)."
        )
        server_startup_delay = 0.5
        port_check_timeout = 2.0
        time.sleep(server_startup_delay)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(port_check_timeout)
                s.connect(("127.0.0.1", port))
            print_with_timestamp(f"Port {port} is open and accepting connections.")
        except OSError as e:
            print_with_timestamp(
                f"WARNING: Could not connect to 127.0.0.1:{port}. "
                f"Relay subprocess may have failed to bind (port in use or firewall). Error: {e}"
            )

    def stop(self):
        """Stop the server so the process can exit (e.g. after Ctrl+C)."""
        relay = getattr(self, "_Server__relay", None)
        if relay is None:
            return
        if hasattr(relay, "_thread"):
            try:
                from twisted.internet import reactor

                reactor.callFromThread(reactor.stop)
            except Exception:
                pass
            relay_thread_join_timeout = 5.0
            relay._thread.join(timeout=relay_thread_join_timeout)


class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """

    def __init__(
        self,
        server_ip=None,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,
        local_com_port=cfg.LOCAL_PORT_TRAINER,
        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME,
        model_path=cfg.MODEL_PATH_TRAINER,
    ):

        self.model_path = model_path
        self.server_ip = server_ip if server_ip is not None else "127.0.0.1"
        self.__endpoint = Endpoint(
            ip_server=self.server_ip,
            port=server_port,
            password=password,
            groups="trainers",
            local_com_port=local_com_port,
            header_size=header_size,
            max_buf_len=max_buf_len,
            security=security,
            keys_dir=keys_dir,
            hostname=hostname,
        )

        print_with_timestamp(f"server IP: {self.server_ip}")

        self.__endpoint.notify(groups={"trainers": -1})

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
        if hasattr(model, "save_to_bytes"):
            weights = model.save_to_bytes()
        else:
            model.save(self.model_path)
            with open(self.model_path, "rb") as f:
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
        self.__endpoint.notify(groups={"trainers": -1})
        if len(res) > 0:
            logger.debug("retrieve_buffer: got {} samples from server", len(res))
        return res


def log_environment_variables():
    """
    add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, "") for k in os.environ.get("LOG_VARIABLES", "").strip().split()}


def load_run_instance(checkpoint_path):
    """
    Default function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    return load(checkpoint_path)


_dump_pids: list[int] = []
_dump_thread: "threading.Thread | None" = None


def dump_run_instance(run_instance, checkpoint_path):
    """
    Default function used to dump trainers to checkpoint path.
    On Unix, uses fork() so the parent returns immediately. On Windows (no fork),
    saves in a background thread so the server does not block and cause socket timeouts.
    """

    def _do_dump():
        try:
            dump(run_instance, checkpoint_path)
        except Exception as e:
            from loguru import logger

            logger.error(f"Error saving checkpoint in background: {e}")

    if hasattr(os, "fork"):
        global _dump_pids
        active_pids = []
        for pid in _dump_pids:
            try:
                pid_ret, _ = os.waitpid(pid, os.WNOHANG)
                if pid_ret == 0:
                    active_pids.append(pid)
            except ChildProcessError:
                pass
        _dump_pids = active_pids

        pid = os.fork()
        if pid == 0:
            try:
                dump(run_instance, checkpoint_path)
            except Exception as e:
                from loguru import logger

                logger.error(f"Error saving checkpoint in child process: {e}")
            finally:
                os._exit(0)
        else:
            _dump_pids.append(pid)
            return
    else:
        global _dump_thread
        dump_thread_join_timeout = 300
        if _dump_thread is not None and _dump_thread.is_alive():
            _dump_thread.join(timeout=dump_thread_join_timeout)
        import collections
        import copy

        snapshot = copy.copy(run_instance)
        snapshot.memory = copy.copy(run_instance.memory)
        snapshot.memory.data = [
            collections.deque(d, maxlen=run_instance.memory.memory_size)
            if isinstance(d, collections.deque)
            else copy.copy(d)
            for d in run_instance.memory.data
        ]
        if hasattr(snapshot.memory, "priorities"):
            snapshot.memory.priorities = copy.copy(run_instance.memory.priorities)
        if hasattr(snapshot.memory, "end_episodes_indices"):
            snapshot.memory.end_episodes_indices = copy.copy(
                run_instance.memory.end_episodes_indices
            )

        def _do_dump_safe():
            try:
                dump(snapshot, checkpoint_path)
            except Exception as e:
                from loguru import logger

                logger.error(f"Error saving checkpoint in background: {e}")

        _dump_thread = threading.Thread(target=_do_dump_safe, daemon=True)
        _dump_thread.start()


def iterate_epochs(
    run_cls,
    interface: TrainerInterface,
    checkpoint_path: str | None,
    dump_run_instance_fn=dump_run_instance,
    load_run_instance_fn=load_run_instance,
    epochs_between_checkpoints=1,
    updater_fn=None,
):
    """
    Main training loop (remote)
    The run_cls instance is saved in checkpoint_path at the end of each epoch
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        logger.debug(f"checkpoint_path: {checkpoint_path}")
        if not exists(checkpoint_path):
            logger.info("=== specification ".ljust(70, "="))
            run_instance = run_cls()
            dump_run_instance_fn(run_instance, checkpoint_path)
            logger.info("")
        else:
            logger.info("Loading checkpoint...")
            t1 = time.time()
            run_instance = load_run_instance_fn(checkpoint_path)
            logger.info(f" Loaded checkpoint in {time.time() - t1} seconds.")
            if updater_fn is not None:
                logger.info("Updating checkpoint...")
                t1 = time.time()
                run_instance = updater_fn(run_instance, run_cls)
                logger.info(f"Checkpoint updated in {time.time() - t1} seconds.")

        while run_instance.epoch < run_instance.epochs:
            yield run_instance.run_epoch(interface=interface)

            if run_instance.epoch % epochs_between_checkpoints == 0:
                logger.info(" saving checkpoint...")
                t1 = time.time()
                dump_run_instance_fn(run_instance, checkpoint_path)
                logger.info(f" saved checkpoint in {time.time() - t1} seconds.")

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def run_with_wandb(
    entity,
    project,
    run_id,
    interface,
    run_cls,
    checkpoint_path: str | None = None,
    dump_run_instance_fn=None,
    load_run_instance_fn=None,
    updater_fn=None,
):
    """
    Main training loop (remote).

    saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)
    import wandb

    cfg.ensure_wandb_api_key()
    logger.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config["environ"] = log_environment_variables()
    hyperparams_dict = cfg.create_config()
    for key, value in hyperparams_dict.items():
        config[key] = value
    resume = bool(checkpoint_path and exists(checkpoint_path))
    wandb_initialized = False
    wandb_max_retries = 10
    wandb_retry_sleep = 10.0
    error_count = 0
    while not wandb_initialized:
        try:
            wandb.init(
                dir=wandb_dir,
                entity=entity,
                project=project,
                id=run_id + " TRAINER",
                resume=resume,
                config=config,
                job_type="trainer",
            )
            wandb_initialized = True

        except Exception as e:
            error_count += 1
            logger.warning(f"wandb error {error_count}: {e}")
            if error_count > wandb_max_retries:
                logger.warning("Could not connect to wandb, aborting.")
                import sys

                sys.exit(1)
            else:
                time.sleep(wandb_retry_sleep)
    for _stats in iterate_epochs(
        run_cls,
        interface,
        checkpoint_path,
        dump_run_instance_fn,
        load_run_instance_fn,
        1,
        updater_fn,
    ):
        # Round-level stats are logged to wandb inside TrainingOffline.run_epoch() with
        # step=total_updates so steps stay monotonically increasing (agent logs per-batch).
        list(_stats)  # consume to drive the epoch


def run(
    interface,
    run_cls,
    checkpoint_path: str | None = None,
    dump_run_instance_fn=None,
    load_run_instance_fn=None,
    updater_fn=None,
):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for _ in iterate_epochs(
        run_cls,
        interface,
        checkpoint_path,
        dump_run_instance_fn,
        load_run_instance_fn,
        1,
        updater_fn,
    ):
        pass


class Trainer:
    """
    Training entity.

    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """

    def __init__(
        self,
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
        dump_run_instance_fn: Callable[..., Any] | None = None,
        load_run_instance_fn: Callable[..., Any] | None = None,
        updater_fn: Callable[..., Any] | None = None,
    ):
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
            checkpoint_path: path for `Trainer` checkpoint (`None` = no checkpointing)
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)
            updater_fn (callable): custom updater (`None` = no updater). If provided, must be a
                function (checkpoint, training_cls) -> updated checkpoint, called after load.
        """
        self.checkpoint_path = checkpoint_path
        self.dump_run_instance_fn = dump_run_instance_fn
        self.load_run_instance_fn = load_run_instance_fn
        self.updater_fn = updater_fn
        self.training_cls = training_cls
        self.interface = TrainerInterface(
            server_ip=server_ip,
            server_port=server_port,
            password=password,
            local_com_port=local_com_port,
            header_size=header_size,
            max_buf_len=max_buf_len,
            security=security,
            keys_dir=keys_dir,
            hostname=hostname,
            model_path=model_path,
        )

    def run(self):
        """
        Runs training.
        """
        run(
            interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn,
        )

    def run_with_wandb(
        self, entity=cfg.WANDB_ENTITY, project=cfg.WANDB_PROJECT, run_id=cfg.WANDB_RUN_ID, key=None
    ):
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
            os.environ["WANDB_API_KEY"] = key
        run_with_wandb(
            entity=entity,
            project=project,
            run_id=run_id,
            interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn,
        )


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
        sample_compressor: Callable[..., Any] | None = None,
        device="cpu",
        max_samples_per_episode=np.inf,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor: Callable[..., Any] | None = None,
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
        hostname=cfg.HOSTNAME,
    ):
        """
        Args:
            env_cls (type): class of the Gymnasium environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): module class for the policy (tmrl.actor.ActorModule subclass)
            sample_compressor (callable): compressor for samples over the Internet; when not `None`,
                must take (prev_act, obs, rew, terminated, truncated, info) and return same order;
                works with a decompression scheme in the Memory class.
            device (str): device on which the policy is running
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): if not None, (obs) -> modified observation
            crc_debug (bool): useful for debugging custom pipelines; leave to False otherwise
            model_path_history (str): path to policy history (omit .tmod); leave default recommended
            model_history (int): save policy every this many new policies (0: not saved)
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
        _obs_dim = (
            sum(math.prod(s.shape or ()) for s in obs_space.spaces)
            if isinstance(obs_space, gymnasium.spaces.Tuple)
            else math.prod(obs_space.shape or ())
        )
        logger.info(
            " Worker env: interface={}, observation_space total_dim={}, "
            "POINTS_NUMBER={}, USE_RNN_MODEL={}",
            cfg_obj.INTERFACE_DISPLAY_NAME,
            _obs_dim,
            cfg_c.POINTS_NUMBER,
            cfg_c.USE_RNN_MODEL,
        )
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.device = device
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to(
            self.device
        )
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logger.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logger.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0
        self.model_cpt = 0

        self.debug_ts_cpt = 0
        self.debug_ts_res_cpt = 0

        self.sde_sample_freq = int(cfg.ALG_CONFIG.get("SDE_SAMPLE_FREQ", 100))
        self.sde_step_counter = 0

        self.start_time = time.time()
        self.server_ip = server_ip if server_ip is not None else "127.0.0.1"

        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            self.__endpoint = Endpoint(
                ip_server=self.server_ip,
                port=server_port,
                password=password,
                groups="workers",
                local_com_port=local_port,
                header_size=header_size,
                max_buf_len=max_buf_len,
                security=security,
                keys_dir=keys_dir,
                hostname=hostname,
                deserializer_mode="synchronous",
            )
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
        try:
            act = self.env.unwrapped.default_action
        except AttributeError:
            act = None

        if hasattr(self.actor, "reset_noise"):
            self.actor.reset_noise(1)
        self.sde_step_counter = 0

        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        terminated, truncated = False, False
        if collect_samples:
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt = 0
                info["crc_sample"] = (obs, act, new_obs, rew, terminated, truncated)
                info["crc_sample_ts"] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(
                    act, new_obs, rew, terminated, truncated, info
                )
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is obs -> act -> (new_obs, rew, terminated, truncated, info).
        In Real-Time RL, act is appended to a buffer that is part of new_obs (real-time delays).

        Args:
            reward_function:
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
        self.sde_step_counter += 1
        if (
            getattr(self.actor, "sde", None) is not None
            and self.sde_step_counter % self.sde_sample_freq == 0
        ):
            self.actor.reset_noise(1)

        act = self.act(obs, test=test)
        new_obs, rew, terminated, truncated, info = self.env.step(act)

        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt += 1
                info["crc_sample"] = (obs, act, new_obs, rew, terminated, truncated)
                info["crc_sample_ts"] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(
                    act, new_obs, rew, terminated, truncated, info
                )
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, rew, terminated, truncated, info

    def collect_train_episode(self, max_samples=None):
        """
        Collects up to `max_samples` training transitions (reset to terminated or truncated).

        Stores the episode and training return in the worker's local Buffer.
        for sending to the `Server`.

        Args:
            max_samples (int): if not terminated after this many steps, reset and set truncated.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, _ = self.reset(collect_samples=True)
        for i in iterator:
            obs, rew, terminated, truncated, _ = self.step(
                obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1
            )
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps
        if self.buffer.memory and self.buffer.stat_train_steps > 0:
            last_info = self.buffer.memory[-1][5]
            if isinstance(last_info, dict) and last_info.get("end_of_track", False):
                time_bonus_scale = float(cfg.REWARD_CONFIG.get("TIME_BONUS_SCALE", 0.0))
                reward_scale = float(cfg.REWARD_CONFIG.get("REWARD_SCALE", 1.0))
                if time_bonus_scale > 0 and reward_scale > 0:
                    self.buffer.apply_speed_bonus(time_bonus_scale * reward_scale)

    def run_episodes(self, max_samples_per_episode=None, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        if max_samples_per_episode is None:
            max_samples_per_episode = self.max_samples_per_episode

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        for _ in iterator:
            self.run_episode(max_samples_per_episode, train=train)

    def _run_deterministic_test_episodes(self, max_samples, n_episodes):
        """Run deterministic eval: competition-style (TMRL test rules) or legacy mean over runs."""
        if cfg_c.BEST_CHECKPOINT_LAP_TIME:
            self._run_competition_style_test_episodes(max_samples, n_episodes)
        else:
            self._run_legacy_deterministic_test_episodes(max_samples, n_episodes)

    def _run_legacy_deterministic_test_episodes(self, max_samples, n_episodes):
        """Mean return/steps; mean finish time over episodes that reached end of track."""
        returns = []
        steps_list = []
        finish_times = []
        for _ in range(n_episodes):
            self.run_episode(max_samples, train=False)
            returns.append(self.buffer.stat_test_return)
            steps_list.append(self.buffer.stat_test_steps)
            if getattr(self.buffer, "stat_test_finished_track", False):
                finish_times.append(self.buffer.stat_test_finish_time)
        self.buffer.stat_test_return = float(np.mean(returns))
        self.buffer.stat_test_steps = float(np.mean(steps_list))
        self.buffer.stat_test_finish_time = float(np.mean(finish_times)) if finish_times else 0.0
        self.buffer.stat_test_finished_count = len(finish_times)
        self.buffer.stat_test_competition_eliminated = False
        self.buffer.stat_test_competition_crashes = n_episodes - len(finish_times)
        if finish_times:
            logger.info(
                "Test runs (epsilon=0) finish times (s): {}  (finished {}/{} episodes)",
                [round(t, 2) for t in finish_times],
                len(finish_times),
                n_episodes,
            )

    def _run_competition_style_test_episodes(self, max_samples, n_episodes):
        """Eval aligned with TMRL competition.

        N attempts, crash discards run, +penalty to next finish.

        After ``COMPETITION_EVAL_MAX_CRASHES`` crashes the eval is eliminated (no valid mean time).
        Mean time is over successful runs only (crashed attempts are excluded from the mean).
        """
        penalty_s = float(cfg_c.COMPETITION_EVAL_CRASH_PENALTY_S)
        max_crashes = int(cfg_c.COMPETITION_EVAL_MAX_CRASHES)
        penalty_carry = 0.0
        crashes = 0
        eliminated = False
        returns: list[float] = []
        steps_list: list[float] = []
        finish_times_adjusted: list[float] = []
        raw_finish_times: list[float] = []

        for _attempt in range(n_episodes):
            if eliminated:
                break
            self.run_episode(max_samples, train=False)
            returns.append(self.buffer.stat_test_return)
            steps_list.append(self.buffer.stat_test_steps)
            finished = bool(getattr(self.buffer, "stat_test_finished_track", False))
            raw_t = float(getattr(self.buffer, "stat_test_finish_time", 0.0))
            if finished and raw_t > 0.0:
                adj = raw_t + penalty_carry
                finish_times_adjusted.append(adj)
                raw_finish_times.append(raw_t)
                penalty_carry = 0.0
            else:
                crashes += 1
                penalty_carry += penalty_s
                if crashes >= max_crashes:
                    eliminated = True
                    break

        self.buffer.stat_test_competition_eliminated = eliminated
        self.buffer.stat_test_competition_crashes = crashes
        self.buffer.stat_test_return = float(np.mean(returns)) if returns else 0.0
        self.buffer.stat_test_steps = float(np.mean(steps_list)) if steps_list else 0.0
        if eliminated or not finish_times_adjusted:
            self.buffer.stat_test_finish_time = 0.0
            self.buffer.stat_test_finished_count = 0
            self.buffer.stat_test_finished_track = False
            logger.info(
                "Competition eval: eliminated={} crashes={}/{}  (no mean lap time)",
                eliminated,
                crashes,
                max_crashes,
            )
        else:
            mean_adj = float(np.mean(finish_times_adjusted))
            self.buffer.stat_test_finish_time = mean_adj
            self.buffer.stat_test_finished_count = len(finish_times_adjusted)
            self.buffer.stat_test_finished_track = len(finish_times_adjusted) > 0
            logger.info(
                "Competition eval: mean lap (penalty-adjusted)={:.2f}s from {}/{} finishes, "
                "crashes={}, raw times (s): {}",
                mean_adj,
                len(finish_times_adjusted),
                n_episodes,
                crashes,
                [round(t, 2) for t in raw_finish_times],
            )

    def run_episode(self, max_samples=None, train=False):
        """
        Collects up to `max_samples` test transitions (reset to terminated or truncated).

        Args:
            max_samples (int): at most this many samples per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        saw_end_of_track = False
        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(
                obs=obs, test=not train, collect_samples=False
            )
            ret += rew
            steps += 1
            if isinstance(info, dict) and info.get("end_of_track"):
                saw_end_of_track = True
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps
        if not train:
            dt = float(cfg.ENV_CONFIG.get("RTGYM_CONFIG", {}).get("time_step_duration", 0.05))
            end_of_track = saw_end_of_track or (
                bool(info.get("end_of_track", False)) if isinstance(info, dict) else False
            )
            self.buffer.stat_test_finish_time = (steps * dt) if end_of_track else 0.0
            self.buffer.stat_test_finished_track = end_of_track

    def run(self, test_episode_interval=0, nb_episodes=np.inf, verbose=True, expert=False):
        """
        Runs the worker for `nb_episodes` episodes.

        Sends episodes to the Server and checks for new weights between episodes.
        For sync/fine-grained sampling use other APIs; for deployment use run_episodes.

        Args:
            test_episode_interval (int): test episode every N train episodes; 0 to disable.
            nb_episodes (int): max train episodes to collect.
            verbose (bool): whether to log INFO messages.
            expert (bool): if True, send samples only, no model updates nor test episodes.
        """

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        if expert:
            if not verbose:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
            else:
                for _ in iterator:
                    print_with_timestamp("collecting expert episode")
                    self.collect_train_episode(self.max_samples_per_episode)
                    print_with_timestamp("copying buffer for sending")
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
        elif not verbose:
            if not test_episode_interval:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
            else:
                n_test_per_eval = getattr(cfg, "RW_TEST_EPISODES_PER_EVAL", 5)
                for episode in iterator:
                    if episode % test_episode_interval == 0 and not self.crc_debug:
                        self._run_deterministic_test_episodes(
                            self.max_samples_per_episode, n_test_per_eval
                        )
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
        else:
            n_test_per_eval = getattr(cfg, "RW_TEST_EPISODES_PER_EVAL", 5)
            for episode in iterator:
                if (
                    test_episode_interval
                    and episode % test_episode_interval == 0
                    and not self.crc_debug
                ):
                    print_with_timestamp(f"running {n_test_per_eval} deterministic test episode(s)")
                    self._run_deterministic_test_episodes(
                        self.max_samples_per_episode, n_test_per_eval
                    )
                print_with_timestamp("collecting train episode")
                self.collect_train_episode(self.max_samples_per_episode)
                print_with_timestamp("copying buffer for sending")
                self.send_and_clear_buffer()
                print_with_timestamp("checking for new weights")
                self.update_actor_weights(verbose=True)

    def run_synchronous(
        self,
        test_episode_interval=0,
        nb_steps=np.inf,
        initial_steps=1,
        max_steps_per_update=np.inf,
        end_episodes=True,
        verbose=False,
    ):
        """
        Collects `nb_steps` steps while synchronizing with the Trainer.

        For traditional (non-real-time) envs that can be stepped fast.
        For rtgym with wait_on_done, set end_episodes to True.

        Note: Does not collect test episodes; use run_episode(train=False) periodically.

        Args:
            test_episode_interval (int): test every N train episodes; 0 to disable.
                Requires end_episodes.
            nb_steps (int): total steps to collect (after initial_steps).
            initial_steps (int): steps before waiting for first model update.
            max_steps_per_update (float): max steps per model from Server (can be non-integer).
            end_episodes (bool): if True, wait for episode end before send/wait; else pause.
            verbose (bool): whether to log INFO messages.
        """

        if verbose:
            logger.info(f"Collecting {initial_steps} initial steps")

        iteration = 0
        done = False
        while iteration < initial_steps:
            steps = 0
            ret = 0.0
            obs, _ = self.reset(collect_samples=True)
            done = False
            iteration += 1
            while not done and (end_episodes or iteration < initial_steps):
                obs, rew, terminated, truncated, _ = self.step(
                    obs=obs,
                    test=False,
                    collect_samples=True,
                    last_step=steps == self.max_samples_per_episode - 1,
                )
                iteration += 1
                steps += 1
                ret += rew
                done = terminated or truncated
            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logger.info("Sending buffer (initial steps)")
            self.send_and_clear_buffer()

        i_model = 1

        ratio = (iteration + 1) / i_model
        while ratio > max_steps_per_update:
            if verbose:
                logger.info(
                    f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates"
                )
            self.send_and_clear_buffer()
            i_model += self.update_actor_weights(verbose=verbose, blocking=True)
            ratio = (iteration + 1) / i_model

        iteration = 0
        episode = 0
        steps = 0
        ret = 0.0

        while iteration < nb_steps:
            if done:
                if (
                    test_episode_interval > 0
                    and episode % test_episode_interval == 0
                    and end_episodes
                ):
                    n_test_per_eval = getattr(cfg, "RW_TEST_EPISODES_PER_EVAL", 5)
                    if verbose:
                        print_with_timestamp(
                            f"running {n_test_per_eval} deterministic test episode(s)"
                        )
                    self._run_deterministic_test_episodes(
                        self.max_samples_per_episode, n_test_per_eval
                    )
                obs, _ = self.reset(collect_samples=True)
                done = False
                iteration += 1
                steps = 0
                ret = 0.0
                episode += 1

            while not done and (end_episodes or ratio <= max_steps_per_update):
                obs, rew, terminated, truncated, _ = self.step(
                    obs=obs,
                    test=False,
                    collect_samples=True,
                    last_step=steps == self.max_samples_per_episode - 1,
                )
                iteration += 1
                steps += 1
                ret += rew

                done = terminated or truncated

                if not end_episodes:
                    ratio = (iteration + 1) / i_model
                    while ratio > max_steps_per_update:
                        if verbose:
                            logger.info(
                                f"Ratio {ratio} > {max_steps_per_update}, sending buffer (no eoe)"
                            )
                        if not done:
                            if verbose:
                                logger.info("Sending buffer (no eoe)")
                            self.send_and_clear_buffer()
                        i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                        ratio = (iteration + 1) / i_model

            if end_episodes:
                ratio = (iteration + 1) / i_model
                while ratio > max_steps_per_update:
                    if verbose:
                        logger.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer (eoe)")
                    if not done:
                        if verbose:
                            logger.info("Sending buffer (eoe)")
                        self.send_and_clear_buffer()
                    i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                    ratio = (iteration + 1) / i_model

            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logger.info(
                    f"Sending buffer - DEBUG ratio {ratio} iteration {iteration} i_model {i_model}"
                )
            self.send_and_clear_buffer()

    def run_env_benchmark(self, nb_steps, test=False, verbose=True):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        The rtgym config must have the "benchmark" option set to True.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
            verbose (bool): whether to log INFO messages
        """
        if nb_steps == np.inf or nb_steps < 0:
            raise RuntimeError(f"Invalid number of steps: {nb_steps}")

        obs, _ = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, _rew, terminated, truncated, _ = self.step(
                obs=obs, test=test, collect_samples=False
            )
            if terminated or truncated:
                break
        res = self.env.unwrapped.benchmarks()
        if verbose:
            print_with_timestamp(f"Benchmark results:\n{res}")
        return res

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self.__endpoint.produce(self.buffer, "trainers")
        self.buffer.clear()

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        weights_list = self.__endpoint.receive_all(blocking=blocking)
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(
                        self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod",
                        "wb",
                    ) as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            if hasattr(self.actor, "load_from_bytes"):
                loaded = self.actor.load_from_bytes(weights, device=self.device)
                if isinstance(loaded, bool):
                    loaded_ok = loaded
                elif loaded is None:
                    loaded_ok = True
                else:
                    # Generic ActorModule implementations may return a new actor instance.
                    self.actor = loaded
                    loaded_ok = True
            else:
                with open(self.model_path, "wb") as f:
                    f.write(weights)
                self.actor = self.actor.load(self.model_path, device=self.device)
                loaded_ok = True
            if verbose:
                if loaded_ok:
                    print_with_timestamp("model weights have been updated")
                else:
                    print_with_timestamp(
                        "model weights NOT applied (shape mismatch); keeping previous weights"
                    )
        return nb_received

    def ignore_actor_weights(self):
        """
        Clears the buffer of weights received from the `Server`.

        This is useful for expert RolloutWorkers, because all RolloutWorkers receive weights.

        Returns:
            int: number of new (ignored) actor models received from the Server.
        """
        weights_list = self.__endpoint.receive_all(blocking=False)
        nb_received = len(weights_list)
        return nb_received
