"""Pydantic models for configuration validation.

This module contains all the configuration models used for type-safe
configuration loading and validation.
"""

from pydantic import BaseModel, Field


class RewardConfig(BaseModel):
    """Reward and termination parameters for the TrackMania reward function."""

    CONSTANT_PENALTY: float = Field(default=0.0, description="Per-step penalty applied each step.")
    CHECK_FORWARD: int = Field(
        default=500, description="Number of trajectory points to look forward for progress."
    )
    CHECK_BACKWARD: int = Field(
        default=10, description="Number of trajectory points to look backward for rewind detection."
    )
    MAX_TIME_NO_PROGRESS_SECONDS: float = Field(
        default=0.0,
        description="If >0: episode ends after this many seconds with no forward progress.",
    )
    MIN_PROGRESS_RATE: float = Field(
        default=0.0,
        description=(
            "If >0: episode ends when progress rate (fraction of track per second) over the last "
            "window is below this."
        ),
    )
    SLOW_PROGRESS_WINDOW_SECONDS: float = Field(
        default=5.0,
        description=(
            "Time window in seconds over which progress rate is measured for slow-progress reset."
        ),
    )
    MIN_STEPS: int = Field(
        default=70,
        description="Minimum steps before failure conditions can trigger (grace period).",
    )
    MAX_STRAY: float = Field(
        default=50.0, description="Max allowed distance from reference trajectory before failure."
    )
    SPEED_SAFE_DEVIATION_RATIO: float = Field(
        default=0.15, description="Speed deviation ratio for safety checks."
    )
    WALL_HUG_SPEED_THRESHOLD: float = Field(
        default=10.0, description="Speed (km/h) below which wall-hug penalty applies."
    )
    WALL_HUG_PENALTY_FACTOR: float = Field(
        default=0.005, description="Scaling factor for wall-hugging penalty."
    )
    REWARD_SCALE: float = Field(
        default=0.6, description="Global scale applied to the total reward."
    )
    SPEED_TERMINAL_SCALE: float = Field(default=0.0, description="Scale for terminal speed bonus.")
    PROJECTED_VELOCITY_SCALE: float = Field(
        default=0.5, description="Weight for reward from velocity along track direction."
    )
    SPEED_REWARD_EXPONENT: float = Field(
        default=1.0,
        description="Exponent for speed-along-track reward (1.0=linear, 2.0=squared).",
    )
    SPEED_REWARD_ALIGNMENT_FLOOR: float = Field(
        default=0.0,
        description="Min alignment (0-1) for speed reward; tolerates drift vs strict heading.",
    )
    DRIFT_REWARD_WEIGHT: float = Field(
        default=0.0, description="Weight for drift/sideslip bonus when slip angle is near optimal."
    )
    DRIFT_OPTIMAL_ANGLE_DEG: float = Field(
        default=12.0, description="Optimal slip angle in degrees for drift reward."
    )
    DRIFT_SIGMA_DEG: float = Field(
        default=8.0, description="Sigma in degrees for Gaussian drift reward."
    )
    DRIFT_THRESHOLD_KMH: float = Field(
        default=80.0, description="Speed (km/h) above which drift reward applies."
    )
    MAX_TRACK_WIDTH: float = Field(
        default=65.0, description="Max distance from track center before boundary penalty."
    )
    BOUNDARY_PENALTY_WEIGHT: float = Field(
        default=4.0, description="Weight for penalty when near track boundary."
    )
    BOUNDARY_CRASH_PENALTY: float = Field(
        default=10.0, description="Penalty when leaving track (crash)."
    )
    REWARD_CLIP_FLOOR: float = Field(default=10.0, description="Floor value when clipping reward.")
    TIME_BONUS_SCALE: float = Field(default=0.0, description="Scale for time-based bonus.")
    CONDITIONAL_PENALTY_WHEN_BRAKING: bool = Field(
        default=False, description="Apply extra penalty only when braking."
    )
    BRAKE_THRESHOLD: float = Field(
        default=0.3, description="Brake input above this triggers conditional penalty."
    )
    TRACK_LOOK_AHEAD_PCT: float = Field(
        default=5.0, description="Percentage of track length used for look-ahead points."
    )
    TRACK_POINT_SPACING_M: float = Field(
        default=2.5, description="Spacing in meters between look-ahead points."
    )
    PROGRESS_REWARD_FULL_LAP: float = Field(
        default=200.0, description="Total reward for completing full lap progress."
    )
    SPEED_REWARD_WEIGHT: float = Field(
        default=0.0, description="Weight for speed-along-track reward component."
    )
    MAX_SPEED_KMH: float = Field(
        default=300.0, description="Reference max speed (km/h) for speed reward scaling."
    )
    CRASH_PENALTY: float = Field(default=2.0, description="Penalty applied on crash event.")
    TRACK_LOCAL_FRAME: bool = Field(
        default=False, description="Use track-aligned local frame for observations."
    )
    DEBUG_REWARD_COMPONENTS: bool = Field(
        default=False, description="Log reward component breakdown when True."
    )
    DEBUG_LOG_INTERVAL: int = Field(default=100, description="Steps between debug reward logs.")
    END_OF_TRACK_REWARD: float = Field(
        default=10.0, description="Reward given when reaching end of track."
    )
    TRACK_CURVATURE_OBS: bool = Field(
        default=False, description="When True, add curvature-at-lookahead to observation."
    )
    MIN_EPISODE_LENGTH_GUARANTEED: int = Field(
        default=100,
        description="Never terminate episode before this many steps (hard grace period).",
    )
    CTE_PENALTY_WEIGHT: float = Field(
        default=0.0,
        description="Weight k for CTE penalty: -k * (d_cte/(w_max/2))^p. 0 disables.",
    )
    CTE_PENALTY_EXPONENT: float = Field(
        default=2.0,
        description="Exponent p for CTE penalty; higher = sharper near-boundary penalty.",
    )
    PROGRESS_MIN_ALIGNMENT: float = Field(
        default=0.0,
        description="Min velocity-track alignment (0-1) for progress reward; 0 disables gating.",
    )
    VELOCITY_ALIGNMENT_REWARD_WEIGHT: float = Field(
        default=0.0,
        description="Weight for explicit alignment bonus (velocity dot track tangent). 0 disables.",
    )
    BARRIER_TOUCH_PENALTY: float = Field(
        default=0.0,
        description=(
            "Penalty when car is within BARRIER_TOUCH_RADIUS of a wall and moving. 0 disables."
        ),
    )
    BARRIER_TOUCH_RADIUS: float = Field(
        default=0.25,
        description="Distance (meters) from barrier polyline within which touch penalty applies. "
        "Left/right track boundaries are the barrier positions.",
    )
    BARRIER_TOUCH_MIN_SPEED_KMH: float = Field(
        default=5.0,
        description="Minimum speed (km/h) for barrier-touch penalty to apply.",
    )
    model_config = {"extra": "allow"}


class EnvConfig(BaseModel):
    """Environment and reward settings from config.json ENV section."""

    RTGYM_INTERFACE: str = Field(
        description="Interface name: e.g. LIDAR, LIDARPROGRESS, TRACKMAP, FULL, MOBILEV3, TQCGRAB."
    )
    SEED: int = Field(0, description="Random seed for environment reproducibility.")
    MAP_NAME: str = Field("", description="Track/map identifier used for paths and rewards.")
    MIN_NB_ZERO_REW_BEFORE_FAILURE: int = Field(
        0, description="Episode ends after this many steps with zero reward."
    )
    MAX_NB_ZERO_REW_BEFORE_FAILURE: int = Field(
        0, description="Upper bound for zero-reward failure count."
    )
    MIN_NB_STEPS_BEFORE_FAILURE: int = Field(
        0, description="Minimum steps before failure condition can trigger."
    )
    OSCILLATION_PERIOD: int = Field(
        0, description="Period for oscillation-based failure detection."
    )
    NB_OBS_FORWARD: int = Field(0, description="Number of forward observations used in reward.")
    CRASH_PENALTY: float = Field(0.0, description="Penalty applied when crash is detected.")
    CRASH_COOLDOWN: int = Field(0, description="Steps to ignore crash after a crash event.")
    CONSTANT_PENALTY: float = Field(0.0, description="Per-step penalty (e.g. -abs(speed)).")
    LAP_REWARD: float = Field(0.0, description="Reward given for completing a lap.")
    LAP_COOLDOWN: int = Field(0, description="Cooldown steps after lap reward.")
    CHECKPOINT_REWARD: float = Field(0.0, description="Reward for passing a checkpoint.")
    END_OF_TRACK_REWARD: float = Field(0.0, description="Reward when reaching end of track.")
    USE_IMAGES: bool = Field(True, description="Whether observation includes image input.")
    SLEEP_TIME_AT_RESET: float = Field(0.0, description="Seconds to sleep after environment reset.")
    IMG_HIST_LEN: int = Field(4, description="Number of image frames in observation history.")
    WINDOW_WIDTH: int = Field(640, description="Game window width in pixels.")
    WINDOW_HEIGHT: int = Field(480, description="Game window height in pixels.")
    IMG_GRAYSCALE: bool = Field(True, description="Use grayscale images when True.")
    IMG_WIDTH: int = Field(64, description="Observation image width after resize.")
    IMG_HEIGHT: int = Field(64, description="Observation image height after resize.")
    LINUX_X_OFFSET: int = Field(64, description="X offset for window capture on Linux.")
    LINUX_Y_OFFSET: int = Field(70, description="Y offset for window capture on Linux.")
    IMG_SCALE_CHECK_ENV: float = Field(1.0, description="Scale factor for image size checks.")
    REWARD_CONFIG: dict = Field(
        default_factory=dict, description="Nested reward parameters; see RewardConfig for keys."
    )
    INIT_GAS_BIAS: float = Field(
        0.0,
        description="Bias for actor output dim 0 (gas) before tanh; e.g. 0.8 => default forward.",
    )
    model_config = {"extra": "allow"}


class DebuggerConfig(BaseModel):
    """Debug and profiling options from config.json DEBUGGER section."""

    DEBUG_MODE: bool = Field(
        default=False, description="Enable debug mode (e.g. anomaly detection, extra logging)."
    )
    CRC_DEBUG: bool = Field(
        default=False, description="Enable CRC checks on samples (pipeline consistency)."
    )
    CRC_DEBUG_SAMPLES: int = Field(
        default=0, description="Number of samples to run CRC checks on when CRC_DEBUG is True."
    )
    PROFILE_TRAINER: bool = Field(
        default=False, description="Profile each epoch with Python profiler."
    )
    WANDB_DEBUG: bool = Field(
        default=True, description="Log extra debug metrics to Weights & Biases."
    )
    PYTORCH_PROFILER: bool = Field(
        default=False, description="Enable PyTorch profiler for training."
    )
    OBSERVATION_BOUNDS_CHECK: bool = Field(
        default=False,
        description="When True, assert batch observations are finite (no NaN/Inf) before forward.",
    )
    model_config = {"extra": "allow"}


class AlgConfig(BaseModel):
    """Algorithm hyperparameters from config.json ALG section (SAC / TQC / REDQSAC)."""

    ALGORITHM: str = Field(
        default="SAC",
        description="One of: SAC, TQC (Truncated Quantile Critic), REDQSAC.",
    )
    LEARN_ENTROPY_COEF: bool = Field(
        default=False, description="If True, use SAC v2 with learnable entropy coefficient."
    )
    LR_ACTOR: float = Field(
        default=1e-5, description="Learning rate for the actor/policy optimizer."
    )
    LR_CRITIC: float = Field(default=5e-5, description="Learning rate for the critic/Q optimizer.")
    LR_ENTROPY: float = Field(
        default=3e-4,
        description="Learning rate for entropy coefficient (when LEARN_ENTROPY_COEF is True).",
    )
    GAMMA: float = Field(default=0.99, description="Discount factor for future rewards.")
    POLYAK: float = Field(
        default=0.995, description="Soft update coefficient for target network (polyak averaging)."
    )
    TARGET_ENTROPY: float = Field(
        default=-0.5, description="Target entropy for automatic alpha tuning (SAC v2)."
    )
    ALPHA: float = Field(
        default=0.01, description="Fixed or initial entropy coefficient (SAC v1 or v2)."
    )
    REDQ_N: int = Field(default=10, description="Number of parallel Q-networks in REDQ.")
    REDQ_M: int = Field(default=2, description="Number of Q-networks to sample for target in REDQ.")
    REDQ_Q_UPDATES_PER_POLICY_UPDATE: int = Field(
        default=20, description="Q updates per policy update (UTD ratio) in REDQ."
    )
    TOP_QUANTILES_TO_DROP: int = Field(
        default=2, description="Number of top quantiles to drop in TQC."
    )
    QUANTILES_NUMBER: int = Field(
        default=1, description="Number of quantiles per Q-network in TQC; must be 1 for SAC."
    )
    N_STEPS: int = Field(default=1, description="N-step return; 1 for single-step TD.")
    R2D2_REWIND: float = Field(default=0.5, description="Rewind ratio for R2D2 replay (when used).")
    R2D2_NUM_SEQUENCES: int = Field(
        default=0,
        description="Num sequences per batch. With R2D2_SEQUENCE_LENGTH>0, use i.i.d. sampling.",
    )
    R2D2_SEQUENCE_LENGTH: int = Field(
        default=0,
        description="Length of each sequence (L). B*L must equal BATCH_SIZE when both >0.",
    )
    R2D2_BURN_IN: int = Field(
        default=40,
        description=(
            "Burn-in prefix length (B) for recurrent replay. "
            "First B steps processed with no_grad to recover hidden state; BPTT on the rest."
        ),
    )
    OPTIMIZER_ACTOR: str = Field(
        default="adam", description="Optimizer for actor: adam, adamw, or sgd."
    )
    OPTIMIZER_CRITIC: str = Field(
        default="adam", description="Optimizer for critic: adam, adamw, or sgd."
    )
    BETAS_ACTOR: list[float] | None = Field(
        default=None, description="Betas for Adam/AdamW actor optimizer."
    )
    BETAS_CRITIC: list[float] | None = Field(
        default=None, description="Betas for Adam/AdamW critic optimizer."
    )
    L2_ACTOR: float | None = Field(default=None, description="Weight decay (L2) for actor.")
    L2_CRITIC: float | None = Field(default=None, description="Weight decay (L2) for critic.")
    NUMBER_OF_POINTS: int = Field(
        default=0, description="Number of track points for reward/observation (when used)."
    )
    POINTS_DISTANCE: float = Field(default=0.0, description="Distance between track points.")
    SPEED_BONUS: float = Field(default=0.0, description="Bonus scale for speed reward component.")
    SPEED_MIN_THRESHOLD: float = Field(default=0.0, description="Speed threshold for reward logic.")
    SPEED_MEDIUM_THRESHOLD: float = Field(
        default=0.0, description="Medium speed threshold for reward logic."
    )
    CLIPPING_WEIGHTS: bool = Field(default=False, description="Whether to clip network weights.")
    CLIP_WEIGHTS_VALUE: float = Field(
        default=1.0, description="Max absolute value for weight clipping."
    )
    ACTOR_WEIGHT_DECAY: float = Field(default=0.0, description="Weight decay for actor optimizer.")
    CRITIC_WEIGHT_DECAY: float = Field(
        default=0.0, description="Weight decay for critic optimizer."
    )
    ADAM_EPS: float = Field(default=1e-8, description="Epsilon for Adam/AdamW optimizers.")
    GRAD_CLIP_ACTOR: float = Field(
        default=1.0, description="Max gradient norm for actor; 0 to disable."
    )
    GRAD_CLIP_CRITIC: float = Field(
        default=1.0, description="Max gradient norm for critic; 0 to disable."
    )
    BACKUP_CLIP_RANGE: float = Field(default=100.0, description="Clip range for TD backup in TQC.")
    REWARD_NORMALIZE_SCALE: float = Field(
        default=1.0,
        description="Scale rewards by 1/this before Bellman backup (e.g. 200). 1.0 = no scaling.",
    )
    DYNAMIC_TRUNCATION_ENABLED: bool = Field(
        default=False,
        description="When True, TQC drops more quantiles when pooled target variance is high.",
    )
    DYNAMIC_TRUNCATION_VARIANCE_PCT: float = Field(
        default=0.9,
        description="Percentile of running variance above which to drop extra quantiles (0-1).",
    )
    VCSE_ENABLED: bool = Field(
        default=False,
        description="VCSE: scale alpha by critic std (more exploration in uncertain states).",
    )
    VCSE_LAMBDA: float = Field(
        default=0.5,
        description="Scaling factor for sigma_Q in VCSE: alpha(s) = alpha_base + lambda * sigma_Q.",
    )
    VCSE_ALPHA_BASE: float = Field(
        default=0.0,
        description="Minimum alpha when VCSE is enabled; added to lambda * sigma_Q.",
    )
    MIXED_PRECISION: bool = Field(
        default=True, description="Use mixed precision (AMP) when available."
    )
    MIXED_PRECISION_DTYPE: str = Field(
        default="bfloat16", description="AMP dtype: bfloat16 or float16."
    )
    USE_SDE: bool = Field(
        default=True,
        description="Use generalized State-Dependent Exploration (gSDE) instead of "
        "independent Gaussian noise. Noise is correlated with latent features.",
    )
    LOG_STD_INIT: float = Field(
        default=-3.0,
        description="Initial value for the gSDE log standard deviation parameter.",
    )
    SDE_CLIP_MEAN: float = Field(
        default=2.0,
        description="Clip actor mean output in [-clip, clip] when using gSDE "
        "to avoid numerical instability. 0 to disable.",
    )
    SDE_SAMPLE_FREQ: int = Field(
        default=100,
        description="Rollout: re-sample gSDE noise every N environment steps. "
        "Lower = more exploration variety, higher = more consistent trajectories.",
    )
    ENTROPY_FLOOR: float = Field(
        default=0.02,
        description="Minimum entropy coefficient (alpha). Prevents entropy collapse.",
    )
    ENTROPY_SCHEDULE: str = Field(
        default="learnable",
        description="Entropy schedule: 'learnable' (SAC auto-tune with floor), "
        "'cosine' (cosine annealing with warm restarts, no learning).",
    )
    ENTROPY_COSINE_T0: int = Field(
        default=300,
        description="First cycle length (in training steps) for cosine entropy schedule.",
    )
    ENTROPY_COSINE_TMULT: float = Field(
        default=1.5,
        description="Cycle length multiplier for cosine entropy schedule.",
    )
    ENTROPY_COSINE_DECAY: float = Field(
        default=0.7,
        description="Amplitude decay per cycle for cosine entropy schedule.",
    )
    PER_TD_BETA: float = Field(
        default=0.4,
        description="Importance-sampling correction exponent for PER-TD (0=no correction, 1=full).",
    )
    model_config = {"extra": "allow"}
