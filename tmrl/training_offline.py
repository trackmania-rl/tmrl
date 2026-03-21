"""Offline training loop: epochs, rounds, buffer retrieval, and model broadcast."""

import datetime
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

import gymnasium
import numpy as np
import torch
from loguru import logger

import tmrl.config.config_objects as cfg_obj
import tmrl.config.constants as cfg
import tmrl.config.paths as cfg_paths
from tmrl.tools.player_runs import (
    align_buffer_observations_to_space,
    filter_buffer_samples_failing_obs_space,
    observation_matches_space,
    poll_player_runs_for_injection,
)
from tmrl.util import pandas_dict, wandb_monotonic_step

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

__docformat__ = "google"

# Keys that must be present for wandb round-level logging (same as networking.run_with_wandb).
_WANDB_ROUND_KEYS = (
    "losses/actor",
    "losses/critic",
    "metrics/return_test",
    "metrics/return_train",
    "metrics/episode_length_test",
    "metrics/episode_length_train",
    "eval/return_deterministic",
    "eval/episode_length_deterministic",
    "eval/finish_time_test_s",
    "eval/finished_track_count_test",
    "eval/competition_eliminated",
    "eval/competition_crashes",
)


def _round_stat_to_wandb_log_dict(round_series) -> dict[str, Any]:
    """Build a sanitized dict from a round stat Series for wandb.log (mirrors networking)."""
    log_dict = round_series.to_dict() if hasattr(round_series, "to_dict") else dict(round_series)
    for k, v in list(log_dict.items()):
        is_invalid = v is None or (
            isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf"))
        )
        if is_invalid:
            log_dict[k] = (
                float("nan")
                if k.startswith("losses/")
                else (
                    0.0
                    if k
                    in (
                        "metrics/return_test",
                        "metrics/return_train",
                        "metrics/episode_length_test",
                        "metrics/episode_length_train",
                        "eval/return_deterministic",
                        "eval/episode_length_deterministic",
                        "eval/finish_time_test_s",
                        "eval/finished_track_count_test",
                        "eval/competition_eliminated",
                        "eval/competition_crashes",
                    )
                    else None
                )
            )
    for key in _WANDB_ROUND_KEYS:
        if key not in log_dict or log_dict[key] is None:
            log_dict[key] = float("nan") if key.startswith("losses/") else 0.0
    return log_dict


def _observation_space_from_sample(observation) -> gymnasium.spaces.Space:
    """Build a gymnasium observation space from a single observation (e.g. tuple of arrays).

    Use this when the replay buffer already has data so the model is built with the same
    observation shape as the data (avoids LayerNorm / backbone shape mismatch).
    """
    if isinstance(observation, (list, tuple)):
        spaces_list = []
        for s in observation:
            arr = np.asarray(s)
            spaces_list.append(
                gymnasium.spaces.Box(
                    low=np.float32(-np.inf),
                    high=np.float32(np.inf),
                    shape=arr.shape,
                    dtype=np.float32,
                )
            )
        return gymnasium.spaces.Tuple(tuple(spaces_list))
    else:
        arr = np.asarray(observation)
        return gymnasium.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=arr.shape,
            dtype=np.float32,
        )


def _observation_dim(space: gymnasium.spaces.Space) -> int:
    """Total dimension of an observation space (Tuple of Box or single Box)."""
    if isinstance(space, gymnasium.spaces.Tuple):
        return sum(math.prod(s.shape or ()) for s in space.spaces)
    return math.prod(space.shape or ())


def _one_obs_from_batch(batch_obs) -> np.ndarray | tuple:
    """Extract a single observation (numpy) from batch observation (tensor or tuple of tensors)."""
    if isinstance(batch_obs, (list, tuple)):
        return tuple(
            cast(np.ndarray, t[0].cpu().numpy() if hasattr(t, "cpu") else np.asarray(t[0]))
            for t in batch_obs
        )
    if hasattr(batch_obs, "cpu"):
        return cast(np.ndarray, batch_obs[0].cpu().numpy())
    return cast(np.ndarray, np.asarray(batch_obs[0]))


def _batch_observation_dim(batch) -> int:
    """Total observation dimension from a training batch (batch[0] = prev_obs)."""
    one_obs = _one_obs_from_batch(batch[0])
    return _observation_dim(_observation_space_from_sample(one_obs))


def _check_observation_integrity(batch) -> None:
    """Assert batch observations are finite (no NaN/Inf) when OBSERVATION_BOUNDS_CHECK is True."""
    if not getattr(cfg, "OBSERVATION_BOUNDS_CHECK", False):
        return
    for name, obs in (("prev_obs", batch[0]), ("next_obs", batch[3])):
        if isinstance(obs, (tuple, list)):
            for i, t in enumerate(obs):
                if (
                    isinstance(t, torch.Tensor)
                    and t.is_floating_point()
                    and (torch.isnan(t).any() or torch.isinf(t).any())
                ):
                    raise ValueError(
                        f"Observation integrity check failed: {name}[{i}] contains NaN or Inf"
                    )
        elif (
            isinstance(obs, torch.Tensor)
            and obs.is_floating_point()
            and (torch.isnan(obs).any() or torch.isinf(obs).any())
        ):
            raise ValueError(f"Observation integrity check failed: {name} contains NaN or Inf")


def _stats_dict_to_numeric(d: dict) -> dict:
    """Convert tensor values in a stats dict to Python scalars so pandas can aggregate."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else float(v.mean().item())
        else:
            out[k] = v
    return out


def _mean_stats_dicts(items: list[dict[str, Any]]) -> dict[str, float]:
    """Fast mean aggregation without pandas DataFrame construction."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in items:
        for k, v in row.items():
            if isinstance(v, Real):
                vf = float(v)
                if vf == vf and vf not in (float("inf"), float("-inf")):
                    sums[k] = sums.get(k, 0.0) + vf
                    counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}


def _concat_batches(batches: list[Any]) -> Any:
    """Concatenate multiple training batches along the batch dimension (dim 0).

    Each batch has the same structure as from memory.sample(): (obs, actions, rewards,
    next_obs, dones, ...) where obs/next_obs may be tuples of tensors. Used when
    BATCHES_PER_STEP > 1 to run multiple R2D2 batches through the model in one step.
    """
    if len(batches) == 1:
        return batches[0]
    n_top = len(batches[0])
    for bi, b in enumerate(batches):
        if len(b) != n_top:
            raise ValueError(
                f"_concat_batches: batch structure mismatch: batch 0 has {n_top} "
                f"elements, batch {bi} has {len(b)}. Ensure all replay samples have "
                "the same format (e.g. same obs tuple length, no mixed worker configs)."
            )
    out: list[Any] = []
    for i in range(n_top):
        elem = batches[0][i]
        if isinstance(elem, (list, tuple)):
            n_inner = min(len(b[i]) for b in batches)
            if n_inner != len(elem):
                raise RuntimeError(
                    f"_concat_batches: tuple length mismatch at index {i}: batch 0 has "
                    f"{len(elem)} elements, min across batches is {n_inner}. Refusing to "
                    "truncate (would silently corrupt training). Ensure all workers use "
                    "the same observation format (e.g. USE_IMAGES) and no corrupted packets. "
                    "Timeouts and validation in retrieve_data() plus interface handling of "
                    "telemetry_invalid/position_patched are the first line of defense against "
                    "corrupted samples entering the replay buffer."
                )
            out.append(
                type(elem)(torch.cat([b[i][j] for b in batches], dim=0) for j in range(n_inner))
            )
        elif isinstance(elem, torch.Tensor):
            out.append(torch.cat([b[i] for b in batches], dim=0))
        elif isinstance(elem, dict):
            merged: dict[str, Any] = {}
            for key in elem:
                vals = [b[i][key] for b in batches]
                if isinstance(vals[0], torch.Tensor):
                    merged[key] = torch.cat(vals, dim=0)
                elif isinstance(vals[0], (bool, int, float)):
                    merged[key] = vals[0]
                else:
                    merged[key] = vals[0]
            out.append(merged)
        else:
            out.append(torch.cat([torch.as_tensor(b[i]) for b in batches], dim=0))
    return type(batches[0])(out)


@dataclass(eq=False)
class TrainingOffline:
    """
    Training wrapper for off-policy algorithms.

    Args:
        env_cls (type): dummy env class for obs/action spaces, or (obs_space, act_space) tuple.
        memory_cls (type): class of the replay memory
        training_agent_cls (type): class of the training agent
        epochs (int): total epochs; agent saved every epoch
        rounds (int): rounds per epoch; statistics every round
        steps (int): training steps per round
        update_model_interval (int): steps between model broadcasts
        update_buffer_interval (int): steps between retrieving buffered samples
        max_training_steps_per_env_step (float): training pauses when above this ratio
        sleep_between_buffer_retrieval_attempts (float): sleep when waiting for samples
        python_profiling (bool): if True, run_epoch is profiled and printed each epoch
        agent_scheduler (callable): if not None, f(Agent, epoch) at start of each epoch
        start_training (int): min samples in replay buffer before starting training
        device (str): device for memory to collate samples
        batches_per_step (int): batches to merge per step (keeps R2D2_NUM_SEQUENCES
            and R2D2_SEQUENCE_LENGTH per batch; better GPU utilization when > 1).
    """

    env_cls: type[Any] | None = None
    memory_cls: type[Any] | None = None
    training_agent_cls: type[Any] | None = None
    epochs: int = 10
    rounds: int = 50
    steps: int = 2000
    update_model_interval: int = 100
    update_buffer_interval: int = 100
    max_training_steps_per_env_step: float = 1.0
    sleep_between_buffer_retrieval_attempts: float = 1.0
    agent_scheduler: Callable[..., Any] | None = None
    start_training: int = 0
    device: str | None = None
    python_profiling: bool = False
    pytorch_profiling: bool = False
    batches_per_step: int = 1
    total_updates = 0

    def __post_init__(self):
        """
        Initializes memory and spaces. The agent is built from actual replay data
        (buffer or first batch), not from env/config, so observation dimension
        always matches the data used for training.
        """
        device = self.device or "cpu"
        self.epoch = 0
        assert self.memory_cls is not None, "memory_cls must be set"
        assert self.training_agent_cls is not None, "training_agent_cls must be set"
        assert self.env_cls is not None, "env_cls must be set"
        self.memory = self.memory_cls(nb_steps=self.steps, device=device)
        if isinstance(self.env_cls, tuple):
            _, action_space = self.env_cls
            self._observation_space_from_env = None
        else:
            with self.env_cls() as env:
                action_space = env.action_space
                self._observation_space_from_env = env.observation_space
                _dim = _observation_dim(env.observation_space)
                logger.info(
                    " Trainer env: interface={}, observation_space total_dim={}",
                    cfg_obj.INTERFACE_DISPLAY_NAME,
                    _dim,
                )
        self._action_space = action_space
        if self._observation_space_from_env is not None and hasattr(
            self.memory, "set_observation_space"
        ):
            self.memory.set_observation_space(self._observation_space_from_env)
        if len(self.memory) > 0:
            if self._observation_space_from_env is not None:
                observation_space = self._observation_space_from_env
                env_dim = _observation_dim(observation_space)
                prev_obs, *_ = self.memory.get_transition(0)
                buffer_dim = _observation_dim(_observation_space_from_sample(prev_obs))
                if buffer_dim != env_dim:
                    logger.warning(
                        " Buffer observation dim ({}) != env dim ({}); clearing buffer so trainer "
                        "matches worker (e.g. TRACK_CURVATURE_OBS changed).",
                        buffer_dim,
                        env_dim,
                    )
                    self.memory.clear()
                dim = _observation_dim(observation_space)
                logger.info(
                    " Building agent from env observation_space (dim={}), trainer matches worker.",
                    dim,
                )
            else:
                prev_obs, *_ = self.memory.get_transition(0)
                observation_space = _observation_space_from_sample(prev_obs)
                dim = _observation_dim(observation_space)
                logger.info(
                    " Inferred observation_space from replay at init (dim={}), building agent.",
                    dim,
                )
            # Enable TF32 for faster matmul on Ampere+ GPUs
            if device.startswith("cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info(" Enabled TF32 for faster CUDA matmul operations")

            self.agent = self.training_agent_cls(
                observation_space=observation_space,
                action_space=action_space,
                device=device,
            )

            # Compile model with torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, "compile") and device.startswith("cuda"):
                try:
                    logger.info(" Compiling model with torch.compile (default mode)...")
                    self.agent.model = torch.compile(self.agent.model, mode="default")
                    self.agent.model_target = torch.compile(self.agent.model_target, mode="default")
                    logger.info(" Model compilation complete")
                except Exception as e:
                    logger.warning(" Model compilation failed (continuing without compile): {}", e)
        else:
            self.agent = None
            logger.info(
                " Replay buffer empty at init; agent will be built from first available data "
                "(env or buffer) so observation_space matches worker."
            )
        self.total_samples = len(self.memory)
        self._injected_player_run_ids: set[str] = set()
        self._best_return_train: float = float("-inf")
        self._best_return_eval: float = float("-inf")
        self._best_mean_lap_time: float = float("inf")
        self._best_epoch: int = -1
        self._perf_acc = {
            "sample_s": 0.0,
            "update_buffer_s": 0.0,
            "train_s": 0.0,
            "wait_ratio_s": 0.0,
            "broadcast_s": 0.0,
            "batches": 0,
        }
        logger.info(f" Initial total_samples:{self.total_samples}")
        if cfg.PLAYER_RUNS_ONLINE_INJECTION:
            from pathlib import Path

            _pr_path = Path(cfg.PLAYER_RUNS_SOURCE_PATH)
            logger.info(
                " Player runs online injection: SOURCE_PATH={} (exists={})",
                cfg.PLAYER_RUNS_SOURCE_PATH,
                _pr_path.exists(),
            )

    def _broadcast_actor_after_rebuild(self, interface) -> None:
        """Push current policy to workers so they match a newly rebuilt agent."""
        if self.agent is None or not hasattr(interface, "broadcast_model"):
            return
        try:
            interface.broadcast_model(self.agent.get_actor())
            logger.info(
                " Broadcast policy after agent rebuild so workers use matching observation shape."
            )
        except Exception as e:
            logger.warning(" Broadcast after agent rebuild failed: {}", e)

    def _ensure_agent_from_data(self, batch=None) -> bool:
        """
        Build or rebuild the agent so observation_space matches the worker (from env_cls)
        or, when env_cls is a tuple, from replay data.
        Using env_cls.observation_space when available ensures trainer and worker
        always use the same model shape (e.g. when TRACK_CURVATURE_OBS or config changes).
        When the current batch has a different obs dim than env (e.g. player runs
        recorded with different config), we rebuild from the batch so training can proceed.

        Returns:
            True if a new agent instance was created, False otherwise.
        """
        observation_space = None
        if self._observation_space_from_env is not None:
            env_dim = _observation_dim(self._observation_space_from_env)
            # If we have a batch and its obs dim differs from env, use batch so we can train
            # on the data we have (e.g. injected player runs with TRACK_CURVATURE_OBS mismatch).
            if batch is not None:
                batch_obs_space = _observation_space_from_sample(_one_obs_from_batch(batch[0]))
                batch_dim = _observation_dim(batch_obs_space)
                if batch_dim != env_dim:
                    # Buffer dim already matches batch (e.g. player runs); skip rebuild each step.
                    if (
                        self.agent is not None
                        and _observation_dim(self.agent.observation_space) == batch_dim
                    ):
                        return False
                    logger.info(
                        " Buffer obs dim ({}) != env ({}); building agent from buffer so "
                        "training can proceed (e.g. player runs from different config).",
                        batch_dim,
                        env_dim,
                    )
                    observation_space = batch_obs_space
                elif (
                    self.agent is not None
                    and _observation_dim(self.agent.observation_space) == env_dim
                ):
                    return False
                else:
                    observation_space = self._observation_space_from_env
            else:
                if (
                    self.agent is not None
                    and _observation_dim(self.agent.observation_space) == env_dim
                ):
                    return False
                if len(self.memory) > 0:
                    prev_obs, *_ = self.memory.get_transition(0)
                    buffer_dim = _observation_dim(_observation_space_from_sample(prev_obs))
                    if buffer_dim != env_dim:
                        logger.warning(
                            " Buffer observation dim ({}) != env dim ({}); "
                            "clearing buffer so trainer matches worker "
                            "(e.g. TRACK_CURVATURE_OBS changed).",
                            buffer_dim,
                            env_dim,
                        )
                        self.memory.clear()
                        self.total_samples = 0
                logger.info(
                    " Building agent from env observation_space (dim={}) "
                    "so trainer matches worker.",
                    env_dim,
                )
                observation_space = self._observation_space_from_env
        elif batch is not None:
            one_obs = _one_obs_from_batch(batch[0])
            batch_obs_space = _observation_space_from_sample(one_obs)
            batch_dim = _observation_dim(batch_obs_space)
            if self.agent is not None:
                current_dim = _observation_dim(self.agent.observation_space)
                if batch_dim == current_dim:
                    return False
                logger.warning(
                    " Observation dim from batch ({}) != agent ({}); rebuilding agent from batch.",
                    batch_dim,
                    current_dim,
                )
            observation_space = batch_obs_space
            logger.info(
                " Building agent from batch (observation dim={}).",
                batch_dim,
            )
        elif self.agent is not None:
            return False
        elif len(self.memory) > 0:
            prev_obs, *_ = self.memory.get_transition(0)
            if isinstance(prev_obs, (list, tuple)):
                one_obs = tuple(
                    (t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)).squeeze()
                    for t in prev_obs
                )
            else:
                arr = prev_obs.cpu().numpy() if hasattr(prev_obs, "cpu") else np.asarray(prev_obs)
                one_obs = arr.squeeze()
            observation_space = _observation_space_from_sample(one_obs)
            dim = _observation_dim(observation_space)
            logger.info(
                " Building agent from memory (observation dim={}).",
                dim,
            )
        if observation_space is None:
            return False
        device = self.device or "cpu"
        # Handle case where checkpoint contains old training_agent_cls with incompatible params
        # (e.g., IQNAgent doesn't accept model_cls/kappa but old checkpoints may have them)
        agent_cls: Any = self.training_agent_cls
        if agent_cls is not None and hasattr(agent_cls, "keywords") and hasattr(agent_cls, "func"):
            import inspect
            from functools import partial as functools_partial

            sig = inspect.signature(agent_cls.func.__init__)
            valid_params = set(sig.parameters.keys())
            # Filter out keywords that the agent class doesn't accept
            # (excluding 'self' which is always in signature but not passed)
            invalid_keys = [k for k in agent_cls.keywords if k not in valid_params]
            if invalid_keys:
                new_keywords = {k: v for k, v in agent_cls.keywords.items() if k in valid_params}
                agent_cls = functools_partial(agent_cls.func, *agent_cls.args, **new_keywords)
        assert agent_cls is not None
        self.agent = agent_cls(
            observation_space=observation_space,
            action_space=self._action_space,
            device=device,
        )
        return True

    def update_buffer(self, interface):
        """
        Updates the memory buffer by appending new data.
        Args: interface (an object with a method retrieve_buffer to get new data)
        Actions:
        Retrieves buffer data from the interface and appends it to the memory.
        Updates the count of total samples.
        Buffers whose observation dim does not match env (e.g. old format from server)
        are discarded so they never enter memory.
        """
        buffer = interface.retrieve_buffer()
        if len(buffer) > 0 and self._observation_space_from_env is not None:
            align_buffer_observations_to_space(buffer, self._observation_space_from_env)
            n_bad = filter_buffer_samples_failing_obs_space(
                buffer, self._observation_space_from_env
            )
            if n_bad:
                logger.warning(
                    " Dropped {} rollout sample(s) that still do not match env observation_space "
                    "(e.g. structurally invalid obs).",
                    n_bad,
                )
            if len(buffer) == 0:
                return
            if not observation_matches_space(buffer.memory[0][1], self._observation_space_from_env):
                logger.warning(
                    " Discarding rollout buffer: first sample still invalid after alignment."
                )
                return
        self.memory.append(buffer)
        self.total_samples += len(buffer)

        if self._ensure_agent_from_data():
            self._broadcast_actor_after_rebuild(interface)

        if not cfg.PLAYER_RUNS_ONLINE_INJECTION:
            return

        demo_buffer, imported_ids, imported_files = poll_player_runs_for_injection(
            source_dir=cfg.PLAYER_RUNS_SOURCE_PATH,
            seen_run_ids=self._injected_player_run_ids,
            max_files=cfg.PLAYER_RUNS_MAX_FILES_PER_UPDATE,
            consume_on_read=cfg.PLAYER_RUNS_CONSUME_ON_READ,
        )
        if len(demo_buffer) > 0:
            n_aligned = align_buffer_observations_to_space(
                demo_buffer, self._observation_space_from_env
            )
            n_demo_bad = filter_buffer_samples_failing_obs_space(
                demo_buffer, self._observation_space_from_env
            )
            if n_demo_bad:
                logger.warning(
                    " Dropped {} player-run sample(s) incompatible with "
                    "observation_space after alignment.",
                    n_demo_bad,
                )
            if n_aligned:
                logger.info(
                    " Normalized {} player-run observation(s) to trainer observation_space "
                    "(dim match worker/replay).",
                    n_aligned,
                )
            if len(demo_buffer) > 0:
                repeat = cfg.PLAYER_RUNS_DEMO_INJECTION_REPEAT
                for _ in range(repeat):
                    self.memory.append(demo_buffer)
                    self.total_samples += len(demo_buffer)
                logger.info(
                    " Injected {} player-run sample(s) from {} file(s), repeat x{} "
                    "(effective: {}). run_ids={}",
                    len(demo_buffer),
                    len(imported_files),
                    repeat,
                    len(demo_buffer) * repeat,
                    sorted(imported_ids),
                )

    def check_ratio(self, interface) -> float:
        """
        Checks the ratio of updates to total samples and waits for new samples if needed.
         Args: interface (an object to retrieve buffer data)
         Actions:
         Ratio of updates to total samples; if over limit or -1, waits for new samples.
        """
        ratio = (
            self.total_updates / self.total_samples
            if self.total_samples > 0.0 and self.total_samples >= self.start_training
            else -1.0
        )
        waited_s = 0.0
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            logger.info(
                " Waiting for new samples (total_samples={}, need >= {} to start)",
                self.total_samples,
                self.start_training,
            )
            wait_attempts = 0
            t_wait_start = time.perf_counter()
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                samples_before = self.total_samples
                self.update_buffer(interface)
                if self.total_samples > samples_before:
                    logger.info(
                        " Received {} samples from server (total: {})",
                        self.total_samples - samples_before,
                        self.total_samples,
                    )
                ratio = (
                    self.total_updates / self.total_samples
                    if self.total_samples > 0.0 and self.total_samples >= self.start_training
                    else -1.0
                )
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    wait_attempts += 1
                    if wait_attempts % 10 == 1 and wait_attempts > 1:
                        logger.info(
                            " Still waiting for samples (total_samples={}, attempt ~{})",
                            self.total_samples,
                            wait_attempts,
                        )
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)
            logger.info(" Resuming training")
            waited_s = time.perf_counter() - t_wait_start
        return waited_s

    def run_round(self, interface, stats_training, t_sample_prev):
        """
        Run one round of training (multiple batches), update buffer and optionally broadcast model.

        Steps:
            1. Every update_buffer_interval steps, pull buffer from interface into replay memory
               and refresh end-of-episode indices and reward sums.
            2. For each batch in memory: call agent.train(), aggregate stats,
            increment total_updates.
            3. Every update_model_interval steps, broadcast current actor weights via interface.
            4. After each batch, call check_ratio to optionally wait for more samples.

        Args:
            interface: Object to retrieve buffer data and broadcast model (e.g. Trainer link).
            stats_training: List to append per-batch training stats (returns, durations, etc.).
            t_sample_prev: Timestamp of previous sample (used for sampling duration in stats).
        """
        num_elements = 5
        step_size = max(1, int(self.steps / (num_elements - 1)))
        batch_index_checkpoints = {i * step_size for i in range(num_elements)}
        n_per_step = max(1, int(self.batches_per_step))
        # Sample synchronously; ThreadPoolExecutor gave no real parallelism due to GIL.
        # For true parallel sampling, use torch.utils.data.DataLoader with num_workers > 0.
        for batch_index in range(self.steps):
            t_sample_start = time.perf_counter()
            batches = [self.memory.sample() for _ in range(n_per_step)]
            batch = _concat_batches(batches)

            # --- FIX: Importance Sampling Weight Normalization ---
            # If the batch contains IS weights (from PER), we MUST normalize by max(w)
            # to prevent gradient spikes as per the Architectural Review.
            if len(batch) >= 7 and isinstance(batch[6], dict):
                info = batch[6]
                if "is_weight" in info:
                    weights = info["is_weight"]
                    # Normalize: w_i = w_i / max(batch_w)
                    max_w = torch.max(weights) + 1e-8
                    info["is_weight"] = weights / max_w

            t_sample = time.time()
            self._perf_acc["sample_s"] += time.perf_counter() - t_sample_start

            if self.total_updates % self.update_buffer_interval == 0:
                t_update_buffer_start = time.perf_counter()
                self.update_buffer(interface)
                self._perf_acc["update_buffer_s"] += time.perf_counter() - t_update_buffer_start

            t_update_buffer = time.time()

            if self.total_updates == 0:
                logger.info("starting training")

            if batch_index in batch_index_checkpoints:
                logger.info(
                    f"batch {batch_index}/{self.steps} finished at: {datetime.datetime.now()}"
                )

            if self._ensure_agent_from_data(batch=batch):
                self._broadcast_actor_after_rebuild(interface)

            # After _ensure_agent_from_data, agent may have been rebuilt from batch when
            # buffer obs dim differed from env (e.g. player runs). Verify batch matches agent.
            if self.agent is not None:
                batch_dim = _batch_observation_dim(batch)
                agent_dim = _observation_dim(self.agent.observation_space)
                if batch_dim != agent_dim:
                    logger.warning(
                        " Batch observation dim ({}) != agent dim ({}). "
                        "Clearing replay buffer and waiting for fresh data.",
                        batch_dim,
                        agent_dim,
                    )
                    self.memory.clear()
                    self.total_samples = 0
                    break

            _check_observation_integrity(batch)
            t_train_start = time.perf_counter()
            stats_training_dict = self.agent.train(batch, self.epoch, batch_index, len(self.memory))
            if not isinstance(stats_training_dict, dict):
                logger.warning(
                    " Agent returned non-dict stats at batch {} (type={}); "
                    "continuing with empty stats.",
                    batch_index,
                    type(stats_training_dict).__name__,
                )
                stats_training_dict = {}
            self._perf_acc["train_s"] += time.perf_counter() - t_train_start

            if "td_errors" in stats_training_dict and "batch_indices" in stats_training_dict:
                bi = stats_training_dict["batch_indices"]
                td = stats_training_dict["td_errors"]
                if hasattr(self.memory, "update_priorities"):
                    indices_tuple = (
                        tuple(bi.tolist()) if hasattr(bi, "tolist") else tuple(int(x) for x in bi)
                    )
                    self.memory.update_priorities(indices_tuple, np.asarray(td))

            # Warn when loss is NaN/inf so it is not silently shown as 0 in logs
            la = stats_training_dict.get("losses/actor")
            lc = stats_training_dict.get("losses/critic")

            def _is_bad(x):
                if x is None:
                    return False
                if isinstance(x, torch.Tensor):
                    return bool(torch.isnan(x).any() or torch.isinf(x).any())
                return math.isnan(x) or math.isinf(x)

            if _is_bad(la) or _is_bad(lc):
                logger.warning(
                    " NaN or inf loss (loss_actor={}, loss_critic={}). "
                    "Try MIXED_PRECISION: false or lower learning rate.",
                    la,
                    lc,
                )

            t_train = time.time()

            stats_training_dict["metrics/return_test"] = self.memory.stat_test_return
            stats_training_dict["metrics/return_train"] = self.memory.stat_train_return
            stats_training_dict["metrics/episode_length_test"] = self.memory.stat_test_steps
            stats_training_dict["metrics/episode_length_train"] = self.memory.stat_train_steps
            # Deterministic eval for separate wandb plots
            stats_training_dict["eval/return_deterministic"] = self.memory.stat_test_return
            stats_training_dict["eval/episode_length_deterministic"] = self.memory.stat_test_steps
            stats_training_dict["eval/finish_time_test_s"] = getattr(
                self.memory, "stat_test_finish_time", 0.0
            )
            stats_training_dict["eval/finished_track_count_test"] = getattr(
                self.memory, "stat_test_finished_count", 0
            )
            stats_training_dict["eval/competition_eliminated"] = float(
                bool(getattr(self.memory, "stat_test_competition_eliminated", False))
            )
            stats_training_dict["eval/competition_crashes"] = float(
                getattr(self.memory, "stat_test_competition_crashes", 0)
            )
            stats_training_dict["timing/sampling_duration"] = t_sample - t_sample_prev
            stats_training_dict["timing/training_step_duration"] = t_train - t_update_buffer
            if hasattr(self.memory, "last_sample_demo_fraction"):
                stats_training_dict["debug/demo_fraction_in_batch"] = float(
                    self.memory.last_sample_demo_fraction
                )
            stats_training += (_stats_dict_to_numeric(stats_training_dict),)
            self.total_updates += 1
            self._perf_acc["batches"] += 1
            if self.total_updates % self.update_model_interval == 0:
                t_broadcast_start = time.perf_counter()
                interface.broadcast_model(self.agent.get_actor())
                self._perf_acc["broadcast_s"] += time.perf_counter() - t_broadcast_start
            self._perf_acc["wait_ratio_s"] += self.check_ratio(interface)

            t_sample_prev = time.time()

    def run_epoch(self, interface):
        """Run one epoch: multiple rounds of training, then increment epoch counter.

        Steps:
            1. Optionally run agent_scheduler(agent, epoch) if set.
            2. For each round: check_ratio (wait for samples if needed), then run_round.
            3. Collect round stats (memory size, round time, idle/update/train times).
            4. If python_profiling is True, run pyinstrument and log profile.
            5. Increment epoch and return list of round stats.

        Args:
            interface: Object to retrieve buffer data and broadcast model.

        Returns:
            List of per-round stat dicts (e.g. round_time, memory_len, return_test).
        """
        stats = []
        if self._ensure_agent_from_data():
            self._broadcast_actor_after_rebuild(interface)

        if (
            self.agent_scheduler is not None
            and callable(self.agent_scheduler)
            and self.agent is not None
        ):
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logger.info(
                f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, "=")
                + f" round {rnd}/{self.rounds} ".ljust(50, "=")
            )
            logger.debug(f"(Training): current memory size:{len(self.memory)}")

            stats_training = []

            t0 = time.time()
            self.check_ratio(interface)
            t1 = time.time()

            if self.python_profiling:
                from pyinstrument import Profiler

                pro = Profiler()
                pro.start()

            t2 = time.time()

            t_sample_prev = t2

            self.run_round(interface, stats_training, t_sample_prev)

            t3 = time.time()

            round_time = t3 - t0
            idle_time = t1 - t0
            update_buf_time = t2 - t1
            train_time = t3 - t2
            logger.debug(
                f"round_time:{round_time}, idle:{idle_time}, update_buf:{update_buf_time}, "
                f"train_time:{train_time}"
            )
            stats += (
                pandas_dict(
                    **{
                        "buffer/memory_len": len(self.memory),
                        "timing/round_time": round_time,
                        "timing/idle_time": idle_time,
                        "step": self.total_updates,
                    },
                    **_mean_stats_dicts(stats_training),
                ),
            )

            # Log round-level stats to wandb here so step is current (avoids "step must be
            # monotonically increasing" when agent logs per-batch).
            if wandb is not None and wandb.run is not None:
                round_log = _round_stat_to_wandb_log_dict(stats[-1])
                step_from_log = round_log.pop("step", None)
                step = int(
                    self.total_updates
                    if step_from_log is None
                    else max(self.total_updates, int(step_from_log))
                )
                step = wandb_monotonic_step(step, wandb.run)
                wandb.log(round_log, step=step)

            logger.info(stats[-1].add_prefix("  ").to_string() + "\n")
            if self._perf_acc["batches"] > 0:
                batches = float(self._perf_acc["batches"])
                logger.info(
                    " Perf avg [ms/batch] sample={:.2f} update_buf={:.2f} train={:.2f} "
                    "broadcast={:.2f} wait_ratio={:.2f} (batches={})",
                    1000.0 * self._perf_acc["sample_s"] / batches,
                    1000.0 * self._perf_acc["update_buffer_s"] / batches,
                    1000.0 * self._perf_acc["train_s"] / batches,
                    1000.0 * self._perf_acc["broadcast_s"] / batches,
                    1000.0 * self._perf_acc["wait_ratio_s"] / batches,
                    self._perf_acc["batches"],
                )

            if self.python_profiling:
                pro.stop()
                logger.info(pro.output_text(unicode=True, color=False, show_all=True))

            # PyTorch profiler: log detailed GPU/CPU profiling to file every epoch
            if self.pytorch_profiling and self.agent is not None:
                self._log_pytorch_profiler_stats()

        self._maybe_save_best_checkpoint(stats)
        self.epoch += 1
        return stats

    def _maybe_save_best_checkpoint(self, stats):
        """Save actor weights when the chosen criterion improves.

        - "eval" + BEST_CHECKPOINT_LAP_TIME: mean lap time (seconds) over rounds that pass
          competition eval (not eliminated, enough finishes); lower is better. If no round
          qualifies, fall back to median metrics/return_test (higher is better).
        - "eval" without lap mode: median metrics/return_test over the epoch.
        - "train": mean metrics/return_train over the epoch (legacy).
        """
        try:
            if self.agent is None:
                return
            if not hasattr(self, "_best_return_eval"):
                self._best_return_eval = float("-inf")
            if not hasattr(self, "_best_mean_lap_time"):
                self._best_mean_lap_time = float("inf")

            def _truthy_eliminated(v) -> bool:
                if v is None or v is False:
                    return False
                if v is True:
                    return True
                try:
                    return float(v) != 0.0
                except (TypeError, ValueError):
                    return bool(v)

            criterion = cfg.BEST_CHECKPOINT_CRITERION
            best_path = cfg_paths.WEIGHTS_FOLDER / "best_actor.pth"

            if criterion == "eval" and cfg.BEST_CHECKPOINT_LAP_TIME:
                min_fin = cfg.BEST_CHECKPOINT_MIN_FINISHES
                if min_fin is None:
                    min_fin = cfg.RW_TEST_EPISODES_PER_EVAL
                min_fin = int(min_fin)

                time_vals: list[float] = []
                returns_all: list[float] = []
                for s in stats:
                    if not hasattr(s, "get"):
                        continue
                    rt = s.get("metrics/return_test", float("nan"))
                    if rt == rt:
                        returns_all.append(float(rt))
                    if _truthy_eliminated(s.get("eval/competition_eliminated", 0.0)):
                        continue
                    ft = float(s.get("eval/finish_time_test_s", 0.0) or 0.0)
                    fc = float(s.get("eval/finished_track_count_test", 0.0) or 0.0)
                    if ft > 0.0 and round(fc) >= min_fin:
                        time_vals.append(ft)

                if time_vals:
                    epoch_mean_time = float(np.mean(time_vals))
                    if epoch_mean_time < self._best_mean_lap_time:
                        self._best_mean_lap_time = epoch_mean_time
                        self._best_epoch = self.epoch
                        torch.save(self.agent.get_actor().state_dict(), str(best_path))
                        logger.info(
                            " New best mean lap (eval)={:.2f}s over {} rounds at "
                            "epoch {} -> saved {}",
                            epoch_mean_time,
                            len(time_vals),
                            self.epoch,
                            best_path,
                        )
                    return

                if not returns_all:
                    return
                agg_ret = float(np.median(returns_all))
                if agg_ret <= self._best_return_eval:
                    return
                self._best_return_eval = agg_ret
                self._best_epoch = self.epoch
                torch.save(self.agent.get_actor().state_dict(), str(best_path))
                logger.info(
                    " New best eval return_test (median fallback)={:.2f} at epoch {} -> saved {}",
                    agg_ret,
                    self.epoch,
                    best_path,
                )
                return

            if criterion == "eval":
                returns = [
                    s.get("metrics/return_test", float("nan")) for s in stats if hasattr(s, "get")
                ]
                valid = [r for r in returns if r == r]
                if not valid:
                    return
                agg_ret = float(np.median(valid))
                if agg_ret <= self._best_return_eval:
                    return
                self._best_return_eval = agg_ret
                self._best_epoch = self.epoch
                torch.save(self.agent.get_actor().state_dict(), str(best_path))
                logger.info(
                    " New best eval return_test (median)={:.2f} at epoch {} -> saved {}",
                    agg_ret,
                    self.epoch,
                    best_path,
                )
            else:
                returns = [
                    s.get("metrics/return_train", float("nan")) for s in stats if hasattr(s, "get")
                ]
                valid = [r for r in returns if r == r]
                if not valid:
                    return
                mean_ret = sum(valid) / len(valid)
                if mean_ret > self._best_return_train and mean_ret > 0:
                    self._best_return_train = mean_ret
                    self._best_epoch = self.epoch
                    best_path = cfg_paths.WEIGHTS_FOLDER / "best_actor.pth"
                    torch.save(self.agent.get_actor().state_dict(), str(best_path))
                    logger.info(
                        " New best return_train={:.2f} at epoch {} -> saved {}",
                        mean_ret,
                        self.epoch,
                        best_path,
                    )
        except Exception as e:
            logger.warning(" Failed to save best checkpoint: {}", e)

    @staticmethod
    def _evt_cuda_time(evt) -> float:
        """Extract CUDA/device time from a profiler event, compatible across PyTorch versions."""
        attrs = (
            "cuda_time_total",
            "self_cuda_time_total",
            "device_time_total",
            "self_device_time_total",
        )
        for attr in attrs:
            val = getattr(evt, attr, None)
            if val is not None:
                return float(val)
        return 0.0

    @staticmethod
    def _evt_cpu_time(evt) -> float:
        for attr in ("cpu_time_total", "self_cpu_time_total"):
            val = getattr(evt, attr, None)
            if val is not None:
                return float(val)
        return 0.0

    def _log_pytorch_profiler_stats(self):
        """Run PyTorch profiler on a single training batch and log results to file."""
        try:
            import json
            from pathlib import Path

            from torch.profiler import ProfilerActivity, profile

            profiler_dir = Path(cfg_paths.WEIGHTS_FOLDER) / "profiler_logs"
            profiler_dir.mkdir(parents=True, exist_ok=True)
            log_file = profiler_dir / f"epoch_{self.epoch}_profile.json"

            batch = self.memory.sample()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                self.agent.train(batch, self.epoch, 0, len(self.memory))

            stats = {
                "epoch": self.epoch,
                "timestamp": datetime.datetime.now().isoformat(),
                "device": str(self.device),
                "batch_size": (
                    self.memory.batch_size if hasattr(self.memory, "batch_size") else "unknown"
                ),
                "memory_len": len(self.memory),
            }

            key_averages = prof.key_averages()
            if len(key_averages) > 0:
                cuda_sorted = sorted(key_averages, key=self._evt_cuda_time, reverse=True)[:20]
                cpu_sorted = sorted(key_averages, key=self._evt_cpu_time, reverse=True)[:20]

                total_cuda = sum(self._evt_cuda_time(e) for e in key_averages)
                total_cpu = sum(self._evt_cpu_time(e) for e in key_averages)

                stats["top_cuda_ops"] = [
                    {
                        "name": evt.key,
                        "cuda_time_ms": round(self._evt_cuda_time(evt) / 1e3, 3),
                        "cuda_time_percent": (
                            round(self._evt_cuda_time(evt) / total_cuda * 100, 2)
                            if total_cuda > 0
                            else 0
                        ),
                        "calls": evt.count,
                    }
                    for evt in cuda_sorted
                ]

                stats["top_cpu_ops"] = [
                    {
                        "name": evt.key,
                        "cpu_time_ms": round(self._evt_cpu_time(evt) / 1e3, 3),
                        "cpu_time_percent": (
                            round(self._evt_cpu_time(evt) / total_cpu * 100, 2)
                            if total_cpu > 0
                            else 0
                        ),
                        "calls": evt.count,
                    }
                    for evt in cpu_sorted
                ]

                stats["total_cuda_time_ms"] = round(total_cuda / 1e3, 3)
                stats["total_cpu_time_ms"] = round(total_cpu / 1e3, 3)
                stats["cuda_cpu_ratio"] = round(total_cuda / total_cpu, 3) if total_cpu > 0 else 0

            with open(log_file, "w") as f:
                json.dump(stats, f, indent=2)

            logger.info(" PyTorch profiler stats saved to {}", log_file)

            if stats.get("top_cuda_ops"):
                logger.info(" Top CUDA operations by time:")
                for i, op in enumerate(stats["top_cuda_ops"][:5], 1):
                    logger.info(
                        "  {}. {}: {:.2f}ms ({:.1f}%)",
                        i,
                        op["name"],
                        op["cuda_time_ms"],
                        op["cuda_time_percent"],
                    )

        except Exception as e:
            logger.warning(" PyTorch profiler failed: {}", e)


class TorchTrainingOffline(TrainingOffline):
    """
    TrainingOffline for trainers based on PyTorch.

    This class implements automatic device selection with PyTorch.
    """

    def __init__(
        self,
        env_cls: type[Any] | None = None,
        memory_cls: type[Any] | None = None,
        training_agent_cls: type[Any] | None = None,
        epochs: int = 10,
        rounds: int = 50,
        steps: int = 2000,
        update_model_interval: int = 100,
        update_buffer_interval: int = 100,
        max_training_steps_per_env_step: float = 1.0,
        sleep_between_buffer_retrieval_attempts: float = 1.0,
        python_profiling: bool = False,
        pytorch_profiling: bool = False,
        agent_scheduler: Callable[..., Any] | None = None,
        start_training: int = 0,
        device: str | None = None,
        batches_per_step: int = 1,
    ):
        """
        Same as TrainingOffline; device=None selects automatically for torch.

        Args:
            env_cls (type): dummy env class or (observation_space, action_space) tuple
            memory_cls (type): replay memory class
            training_agent_cls (type): training agent class
            epochs (int): total epochs
            rounds (int): rounds per epoch
            steps (int): training steps per round
            update_model_interval (int): steps between model broadcasts
            update_buffer_interval (int): steps between retrieving buffered samples
            max_training_steps_per_env_step (float): pause training when above this ratio
            sleep_between_buffer_retrieval_attempts (float): sleep when waiting for samples
            python_profiling (bool): profile run_epoch and print at end of each epoch
            pytorch_profiling (bool): profile PyTorch operations and log to file
            agent_scheduler (callable): if not None, f(Agent, epoch) at start of each epoch
            start_training (int): min samples in replay buffer before training
            device (str): device for memory (None = auto)
            batches_per_step (int): number of batches to merge per training step
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            env_cls,
            memory_cls,
            training_agent_cls,
            epochs,
            rounds,
            steps,
            update_model_interval,
            update_buffer_interval,
            max_training_steps_per_env_step,
            sleep_between_buffer_retrieval_attempts,
            agent_scheduler,
            start_training,
            device,
            python_profiling,
            pytorch_profiling,
            batches_per_step,
        )
