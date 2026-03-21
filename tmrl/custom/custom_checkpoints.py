import itertools
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.optim import Adam

import tmrl.config as cfg
from tmrl.util import dump, load


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
    tar_path = str(parent_path / "dataset.tar")
    dataset_path = str(cfg.DATASET_PATH)
    logger.debug(f" load: tar_path :{tar_path}")
    logger.debug(f" load: dataset_path :{dataset_path}")
    with tarfile.open(tar_path, "r") as t:
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
    tar_path = str(parent_path / "dataset.tar")
    dataset_path = str(cfg.DATASET_PATH)
    logger.debug(f" dump: tar_path :{tar_path}")
    logger.debug(f" dump: dataset_path :{dataset_path}")
    with tarfile.open(tar_path, "w") as tar_handle:
        for root, _dirs, files in os.walk(dataset_path):
            for file in files:
                tar_handle.add(os.path.join(root, file), arcname=file)
    dump(run_instance, checkpoint_path)


def update_memory(run_instance):
    """
    Updates memory-related config in a run instance if they differ from current settings.

    Args:
        run_instance: Instance of a run or training session with config and memory settings.
    Returns:
        The updated run_instance, or the original if no modifications were needed.
    """
    steps = cfg.MODEL_CONFIG["TRAINING_STEPS_PER_ROUND"]
    memory_size = cfg.MODEL_CONFIG["MEMORY_SIZE"]
    batch_size = cfg.MODEL_CONFIG["BATCH_SIZE"]
    if (
        run_instance.steps != steps
        or run_instance.memory.batch_size != batch_size
        or run_instance.memory.memory_size != memory_size
    ):
        run_instance.steps = steps
        run_instance.memory.nb_steps = steps
        run_instance.memory.batch_size = batch_size
        run_instance.memory.memory_size = memory_size
        logger.info(
            f"Memory updated with steps:{steps}, batch size:{batch_size}, "
            f"memory size:{memory_size}."
        )
    return run_instance


def update_run_instance(run_instance, training_cls):
    """
    Updates the checkpoint after loading with compatible values from config.json

    Args:
        run_instance: the instance of the checkpoint to update
        training_cls: partially instantiated class of a new checkpoint (to replace if needed)

    Returns:
        run_instance: the updated checkpoint
    """
    # check whether we should start a new experiment entirely and keep only the memory:
    if cfg.TMRL_CONFIG.get("RESET_TRAINING"):
        new_run_instance = training_cls()
        new_run_instance.memory = run_instance.memory
        new_run_instance = update_memory(new_run_instance)
        new_run_instance.total_samples = len(new_run_instance.memory)
        return new_run_instance

    # update training Agent:
    alg_config = cfg.TMRL_CONFIG["ALG"]
    alg_name = alg_config["ALGORITHM"]
    if alg_name not in ["SAC", "REDQSAC", "TQC"]:
        run_instance = update_memory(run_instance)
        return run_instance

    if alg_name in ["SAC", "REDQSAC", "TQC"]:
        lr_actor = alg_config["LR_ACTOR"]
        lr_critic = alg_config["LR_CRITIC"]
        lr_entropy = alg_config["LR_ENTROPY"]
        gamma = alg_config["GAMMA"]
        polyak = alg_config["POLYAK"]
        learn_entropy_coef = alg_config["LEARN_ENTROPY_COEF"]
        target_entropy = alg_config["TARGET_ENTROPY"]
        alpha = alg_config["ALPHA"]

        agent = getattr(run_instance, "agent", None)
        if agent is None:
            logger.debug(
                "Checkpoint has no agent (e.g. empty buffer at save); skipping agent config update."
            )
        else:
            if alg_name in ("SAC", "TQC"):
                if run_instance.agent.lr_actor != lr_actor:
                    old = run_instance.agent.lr_actor
                    run_instance.agent.lr_actor = lr_actor
                    run_instance.agent.actor_optimizer = Adam(
                        run_instance.agent.model.actor.parameters(), lr=lr_actor
                    )
                    logger.info(
                        f"Actor optimizer reinitialized with new lr: {lr_actor} (old lr: {old})."
                    )

                if run_instance.agent.lr_critic != lr_critic:
                    old = run_instance.agent.lr_critic
                    run_instance.agent.lr_critic = lr_critic
                    run_instance.agent.critic_optimizer = Adam(
                        itertools.chain(
                            run_instance.agent.model.q1.parameters(),
                            run_instance.agent.model.q2.parameters(),
                        ),
                        lr=lr_critic,
                    )
                    logger.info(
                        f"Critic optimizer reinitialized with new lr: {lr_critic} (old lr: {old})."
                    )

            if run_instance.agent.learn_entropy_coef != learn_entropy_coef:
                logger.warning("Cannot switch entropy learning.")

            if run_instance.agent.lr_entropy != lr_entropy or run_instance.agent.alpha != alpha:
                run_instance.agent.lr_entropy = lr_entropy
                run_instance.agent.alpha = alpha
                device = run_instance.device or ("cuda" if torch.cuda.is_available() else "cpu")
                if run_instance.agent.learn_entropy_coef:
                    run_instance.agent.log_alpha = (
                        torch.log(torch.ones(1) * run_instance.agent.alpha)
                        .to(device)
                        .requires_grad_(True)
                    )
                    run_instance.agent.alpha_optimizer = Adam(
                        [run_instance.agent.log_alpha], lr=lr_entropy
                    )
                    logger.info("Entropy optimizer reinitialized.")
                else:
                    run_instance.agent.alpha_t = torch.tensor(float(run_instance.agent.alpha)).to(
                        device
                    )
                    logger.info(f"Alpha changed to {alpha}.")

            if run_instance.agent.gamma != gamma:
                old = run_instance.agent.gamma
                run_instance.agent.gamma = gamma
                logger.info(f"Gamma coefficient changed to {gamma} (old: {old}).")

            if run_instance.agent.polyak != polyak:
                old = run_instance.agent.polyak
                run_instance.agent.polyak = polyak
                logger.info(f"Polyak coefficient changed to {polyak} (old: {old}).")

            if target_entropy is None:  # automatic entropy coefficient
                action_space = run_instance.agent.action_space
                run_instance.agent.target_entropy = -np.prod(
                    action_space.shape
                )  # .astype(np.float32)
            else:
                run_instance.agent.target_entropy = float(target_entropy)
            logger.info(f"Target entropy: {run_instance.agent.target_entropy}.")

            if alg_name == "REDQSAC":
                m = alg_config["REDQ_M"]
                q_updates_per_policy_update = alg_config["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]

                if run_instance.agent.q_updates_per_policy_update != q_updates_per_policy_update:
                    old = run_instance.agent.q_updates_per_policy_update
                    run_instance.agent.q_updates_per_policy_update = q_updates_per_policy_update
                    logger.info(
                        f"Q update ratio switched to {q_updates_per_policy_update} (old: {old})."
                    )

                if run_instance.agent.m != m:
                    old = run_instance.agent.m
                    run_instance.agent.m = m
                    logger.info(f"M switched to {m} (old: {old}).")

    epochs = cfg.MODEL_CONFIG["MAX_EPOCHS"]
    rounds = cfg.MODEL_CONFIG["ROUNDS_PER_EPOCH"]
    update_model_interval = cfg.MODEL_CONFIG["UPDATE_MODEL_INTERVAL"]
    update_buffer_interval = cfg.MODEL_CONFIG["UPDATE_BUFFER_INTERVAL"]
    max_training_steps_per_env_step = cfg.MODEL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
    profiling = cfg.PROFILE_TRAINER
    start_training = cfg.MODEL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

    if run_instance.epochs != epochs:
        old = run_instance.epochs
        run_instance.epochs = epochs
        logger.info(f"Max epochs changed to {epochs} (old: {old}).")

    if run_instance.rounds != rounds:
        old = run_instance.rounds
        run_instance.rounds = rounds
        logger.info(f"Rounds per epoch changed to {rounds} (old: {old}).")

    if run_instance.update_model_interval != update_model_interval:
        old = run_instance.update_model_interval
        run_instance.update_model_interval = update_model_interval
        logger.info(f"Model update interval changed to {update_model_interval} (old: {old}).")

    if run_instance.update_buffer_interval != update_buffer_interval:
        old = run_instance.update_buffer_interval
        run_instance.update_buffer_interval = update_buffer_interval
        logger.info(f"Buffer update interval changed to {update_buffer_interval} (old: {old}).")

    if run_instance.max_training_steps_per_env_step != max_training_steps_per_env_step:
        old = run_instance.max_training_steps_per_env_step
        run_instance.max_training_steps_per_env_step = max_training_steps_per_env_step
        logger.info(
            f"Max train/env step ratio changed to {max_training_steps_per_env_step} (old: {old})."
        )

    if run_instance.python_profiling != profiling:
        old = run_instance.python_profiling
        run_instance.python_profiling = profiling
        logger.info(f"Profiling witched to {profiling} (old: {old}).")

    if run_instance.start_training != start_training:
        old = run_instance.start_training
        run_instance.start_training = start_training
        logger.info(
            f"Number of environment steps before training changed to {start_training} (old: {old})."
        )

    run_instance = update_memory(run_instance)

    return run_instance
