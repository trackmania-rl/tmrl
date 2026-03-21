"""Record one or more episodes as player-run files for later replay import."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from loguru import logger

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.custom.tm.utils.control_keyboard import is_del_pressed
from tmrl.envs import GenericGymEnv
from tmrl.tools.player_runs import align_observation_to_space, save_player_run
from tmrl.util import partial


def _extract_human_action_from_obs_tqcgrab(obs):
    """Extract [gas, brake, steer] from TQCGRAB observation (indices 5,6,7 = steer, gas, brake)."""
    gas = float(np.asarray(obs[6]).flat[0])
    brake = float(np.asarray(obs[7]).flat[0])
    steer = float(np.asarray(obs[5]).flat[0])
    return np.array([gas, brake, steer], dtype=np.float32)


def _collect_human_episode(env, max_samples, obs_preprocessor, crc_debug):
    """Collect one episode using human control (neutral sent so human drives) and Del to end."""
    neutral_action = np.zeros(3, dtype=np.float32)
    buffer_memory = []
    ret = 0.0
    steps = 0

    obs, info = env.reset()
    if obs_preprocessor is not None:
        obs = obs_preprocessor(obs)

    iterator = range(max_samples) if max_samples != np.inf else itertools.count()
    for i in iterator:
        new_obs, rew, terminated, truncated, info = env.step(neutral_action)
        if obs_preprocessor is not None:
            new_obs = obs_preprocessor(new_obs)

        if i == max_samples - 1 and not terminated:
            truncated = True

        act_for_sample = _extract_human_action_from_obs_tqcgrab(new_obs)
        if is_del_pressed():
            truncated = True

        if crc_debug:
            info = dict(info)
            info["crc_sample"] = (obs, act_for_sample, new_obs, rew, terminated, truncated)
            info["crc_sample_ts"] = (0, steps)
        sample = (act_for_sample, new_obs, rew, terminated, truncated, info)
        buffer_memory.append(sample)

        ret += rew
        steps += 1
        obs = new_obs

        if terminated or truncated:
            break

    return buffer_memory, ret, steps


def record_episode(
    *,
    nb_episodes: int = 1,
    output_dir: str | None = None,
    max_samples_per_episode: int | None = None,
    save_replays: bool = False,
) -> list[Path]:
    """Collect episodes and save them as standalone player-run files.

    Uses human control (no model): sends neutral (0,0,0) so you drive with a physical
    gamepad. Press Del to end the current episode early.
    """
    if nb_episodes <= 0:
        raise ValueError("nb_episodes must be > 0")

    if not cfg.PRAGMA_TQC_GRAB:
        raise NotImplementedError(
            "Human recording is only supported for TQCGRAB interface. "
            "Set RTGYM_INTERFACE to TQCGRAB in config."
        )

    env_config = cfg_obj.CONFIG_DICT.copy()
    interface_kwargs = dict(env_config.get("interface_kwargs") or {})
    interface_kwargs["record_human"] = True
    if save_replays:
        interface_kwargs["save_replays"] = True
    env_config["interface_kwargs"] = interface_kwargs
    # Ensure record_human reaches the interface (rtgym may not merge interface_kwargs)
    _int = env_config["interface"]
    if hasattr(_int, "func") and hasattr(_int, "keywords"):
        env_config["interface"] = partial(_int.func, record_human=True, **_int.keywords)

    max_samples = (
        int(max_samples_per_episode)
        if max_samples_per_episode is not None
        else cfg.RW_MAX_SAMPLES_PER_EPISODE
    )

    env_cls = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": env_config})

    saved_paths: list[Path] = []
    with env_cls() as env:
        for ep in range(nb_episodes):
            input(
                f"Press Enter to start episode {ep + 1}/{nb_episodes} "
                "(be IN MAP with car on track) ... "
            )
            logger.info(
                "Recording episode {}/{} (human control; press Del to end) ...",
                ep + 1,
                nb_episodes,
            )
            samples, ep_return, ep_steps = _collect_human_episode(
                env,
                max_samples=max_samples,
                obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                crc_debug=cfg.CRC_DEBUG,
            )

            obs_space = env.observation_space
            samples = [
                (
                    s[0],
                    align_observation_to_space(s[1], obs_space),
                    s[2],
                    s[3],
                    s[4],
                    s[5],
                )
                for s in samples
            ]

            metadata = {
                "episode_index": ep,
                "episode_return": float(ep_return),
                "episode_steps": int(ep_steps),
                "map_name": cfg.MAP_NAME,
                "run_name": cfg.RUN_NAME,
                "memory_class": cfg_obj.MEMORY.func.__name__
                if hasattr(cfg_obj.MEMORY, "func")
                else "unknown",
            }
            out_path = save_player_run(samples, output_dir=output_dir, metadata=metadata)
            saved_paths.append(out_path)
            logger.info(
                "Saved {} samples to '{}'. return={} steps={}",
                len(samples),
                out_path,
                metadata["episode_return"],
                metadata["episode_steps"],
            )

    logger.info("Recorded {} episode file(s).", len(saved_paths))
    return saved_paths
