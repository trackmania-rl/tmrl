# standard library imports
import os
import tarfile
from pathlib import Path

# local imports
from tmrl.config import config_constants as cfg
from tmrl.util import dump, load
import logging

def load_run_instance_images_dataset(checkpoint_path):
    """
    function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    chk_path = Path(checkpoint_path)
    parent_path = chk_path.parent.absolute()
    tar_path = str(parent_path / 'dataset.tar')
    dataset_path = str(cfg.DATASET_PATH)
    logging.debug(f" load: tar_path :{tar_path}")
    logging.debug(f" load: dataset_path :{dataset_path}")
    with tarfile.open(tar_path, 'r') as t:
        t.extractall(dataset_path)
    return load(checkpoint_path)


def dump_run_instance_images_dataset(run_instance, checkpoint_path):
    """
    function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    chk_path = Path(checkpoint_path)
    parent_path = chk_path.parent.absolute()
    tar_path = str(parent_path / 'dataset.tar')
    dataset_path = str(cfg.DATASET_PATH)
    logging.debug(f" dump: tar_path :{tar_path}")
    logging.debug(f" dump: dataset_path :{dataset_path}")
    with tarfile.open(tar_path, 'w') as tar_handle:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                tar_handle.add(os.path.join(root, file), arcname=file)
    dump(run_instance, checkpoint_path)
