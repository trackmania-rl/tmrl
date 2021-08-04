# logger (basicConfig must be called before importing anything)
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# standard library imports
import atexit
import gc
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from os.path import exists
from random import randrange
from tempfile import mkdtemp

# third-party imports
import pandas as pd
import yaml

# local imports
import tmrl.sac
from tmrl.networking import TrainerInterface
from tmrl.training_offline import TrainingOffline
# from tmrl.envs import AvenueEnv
from tmrl.util import (dump, git_info, load, load_json, partial, partial_from_dict,
                       partial_to_dict, save_json)

__version__ = "0.9"


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


def iterate_epochs_tm(run_cls: TrainingOffline,
                      interface: TrainerInterface,
                      checkpoint_path: str,
                      dump_run_instance_fn=dump_run_instance,
                      load_run_instance_fn=load_run_instance,
                      epochs_between_checkpoints=1):
    """
    Main training loop for trackmania (remote)
    The TrainingOffline object is saved in checkpoint_path at the end of each epoch
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        if not exists(checkpoint_path):
            logging.info(f"=== specification ".ljust(70, "="))
            # logging.debug(f"{partial_to_dict(run_cls)}")
            # logging.info(yaml.dump(partial_to_dict(run_cls), indent=3, default_flow_style=False, sort_keys=False), end="")
            run_instance = run_cls()
            dump_run_instance_fn(run_instance, checkpoint_path)
            logging.info(f"")
        else:
            logging.info(f" Loading checkpoint...")
            t1 = time.time()
            run_instance = load_run_instance_fn(checkpoint_path)
            logging.info(f" Loaded checkpoint in {time.time() - t1} seconds.")

        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch(interface=interface)  # yield stats data frame (this makes this function a generator)
            # logging.info(f"")
            if run_instance.epoch % epochs_between_checkpoints == 0:
                logging.info(f" saving checkpoint...")
                t1 = time.time()
                dump_run_instance_fn(run_instance, checkpoint_path)
                logging.info(f" saved checkpoint in {time.time() - t1} seconds.")
                # # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
                # del run_instance
                # gc.collect()  # garbage collection
                # run_instance = load_run_instance_fn(checkpoint_path)

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def run_wandb_tm(entity, project, run_id, interface, run_cls: type = TrainingOffline, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None):
    """
    trackmania main (remote)
    run and save config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    # third-party imports
    import wandb
    logging.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config['seed'] = config['seed'] or randrange(1, 1000000)  # if seed == 0 replace with random
    config['environ'] = log_environment_variables()
    #config['git'] = git_info()  # TODO: check this for bugs
    resume = checkpoint_path and exists(checkpoint_path)
    wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
    # logging.info(config)
    # exit()
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run_tm(interface, run_cls: type = TrainingOffline, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None):
    """
    trackmania main (remote)
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn):
        pass
