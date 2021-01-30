from dataclasses import dataclass
import atexit
import gc
import json
import os
import shutil
import tempfile
import time
from os.path import exists
from random import randrange
from tempfile import mkdtemp
import pandas as pd
import yaml

# from tmrl.envs import AvenueEnv
from tmrl.util import partial, save_json, partial_to_dict, partial_from_dict, load_json, dump, load, git_info
from tmrl.training_offline import TrainingOffline
import tmrl.sac
from tmrl.networking import TrainerInterface


def iterate_episodes(run_cls: type = TrainingOffline, checkpoint_path: str = None):
    """Generator [1] yielding episode statistics (list of pd.Series) while running and checkpointing
    - run_cls: can by any callable that outputs an appropriate run object (e.g. has a 'run_epoch' method)

    [1] https://docs.python.org/3/howto/functional.html#generators
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        if not exists(checkpoint_path):
            print("=== specification ".ljust(70, "="))
            print(f"DEBUG:{partial_to_dict(run_cls)}")
            # print(yaml.dump(partial_to_dict(run_cls), indent=3, default_flow_style=False, sort_keys=False), end="")
            run_instance = run_cls()
            dump(run_instance, checkpoint_path)
            print("")
        else:
            print("\ncontinuing...\n")

        run_instance = load(checkpoint_path)
        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch()  # yield stats data frame (this makes this function a generator)
            print("")
            dump(run_instance, checkpoint_path)

            # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
            del run_instance
            gc.collect()
            run_instance = load(checkpoint_path)

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def log_environment_variables():
    """add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}


def run(run_cls: type = TrainingOffline, checkpoint_path: str = None):
    list(iterate_episodes(run_cls, checkpoint_path))


def run_wandb(entity, project, run_id, run_cls: type = TrainingOffline, checkpoint_path: str = None):
    """run and save config and stats to https://wandb.com"""
    wandb_dir = mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    import wandb
    config = partial_to_dict(run_cls)
    config['seed'] = config['seed'] or randrange(1, 1000000)  # if seed == 0 replace with random
    config['environ'] = log_environment_variables()
    config['git'] = git_info()
    resume = checkpoint_path and exists(checkpoint_path)
    wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
    for stats in iterate_episodes(run_cls, checkpoint_path):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run_fs(path: str, run_cls: type = TrainingOffline):
    """run and save config and stats to `path` (with pickle)"""
    if not exists(path):
        os.mkdir(path)
    save_json(partial_to_dict(run_cls), path + '/spec.json')
    if not exists(path + '/stats'):
        dump(pd.DataFrame(), path + '/stats')
    for stats in iterate_episodes(run_cls, path + '/state'):
        dump(load(path + '/stats').append(stats, ignore_index=True),
             path + '/stats')  # concat with stats from previous episodes


# TRACKMANIA REMOTE TRAINING: =====================================================


def iterate_epochs_tm(run_cls: TrainingOffline, interface: TrainerInterface, checkpoint_path: str):
    """
    Main training loop for trackmania (remote)
    The TrainingOffline object is saved in checkpoint_path every checkpoint_interval epochs
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator [1] yielding episode statistics (list of pd.Series) while running and checkpointing
    [1] https://docs.python.org/3/howto/functional.html#generators
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        if not exists(checkpoint_path):
            print("=== specification ".ljust(70, "="))
            # print(f"DEBUG:{partial_to_dict(run_cls)}")
            # print(yaml.dump(partial_to_dict(run_cls), indent=3, default_flow_style=False, sort_keys=False), end="")
            run_instance = run_cls()
            dump(run_instance, checkpoint_path)
            print("")
        else:
            print("\ncontinuing...\n")

        run_instance = load(checkpoint_path)
        cpt = 0
        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch(interface=interface)  # yield stats data frame (this makes this function a generator)
            # print("")
            dump(run_instance, checkpoint_path)

            # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
            del run_instance
            gc.collect()
            run_instance = load(checkpoint_path)
            cpt += 1

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def run_wandb_tm(entity, project, run_id, interface, run_cls: type = TrainingOffline, checkpoint_path: str = None):
    """
    trackmania main (remote)
    run and save config and stats to https://wandb.com
    """
    wandb_dir = mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    import wandb
    print(f"DEBUG: run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config['seed'] = config['seed'] or randrange(1, 1000000)  # if seed == 0 replace with random
    config['environ'] = log_environment_variables()
    #config['git'] = git_info()  # TODO: check this for bugs
    resume = checkpoint_path and exists(checkpoint_path)
    wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
    # print(config)
    # exit()
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run_tm(interface, run_cls: type = TrainingOffline, checkpoint_path: str = None):
    """
    trackmania main (remote)
    """
    for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path):
        pass
