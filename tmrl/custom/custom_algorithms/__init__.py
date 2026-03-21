"""Training agents for TMRL: SAC, REDQ-SAC, SAC (config-driven), TQC, IQN, and SD-SAC.

SAC and REDQ-SAC are Soft Actor-Critic variants. TQC implements distributional
critics with truncation to control overestimation bias (Kuznetsov et al., 2020).
IQN implements Implicit Quantile Networks with Double DQN and Dueling heads
for discrete-action control (Dabney et al., 2018).
SD-SAC implements Stable Discrete SAC with Double Avg Q, Q-clip, and Entropy
Penalty (Zhou et al., TMLR 2024).
"""

from tmrl.custom.custom_algorithms._common import (
    _amp_dtype,
    _amp_enabled,
    _compute_n_step_return_and_bootstrap_mask,
    _tensor_to_scalar,
    set_seed,
)
from tmrl.custom.custom_algorithms.iqn import IQNAgent
from tmrl.custom.custom_algorithms.redq_sac import REDQSACAgent
from tmrl.custom.custom_algorithms.sac import SpinupSacAgent
from tmrl.custom.custom_algorithms.sac_config import SpinupSacAgentConfig
from tmrl.custom.custom_algorithms.sdsac import SDSACAgent
from tmrl.custom.custom_algorithms.tqc import TQCAgent

__all__ = [
    "IQNAgent",
    "REDQSACAgent",
    "SDSACAgent",
    "SpinupSacAgent",
    "SpinupSacAgentConfig",
    "TQCAgent",
    "_amp_dtype",
    "_amp_enabled",
    "_compute_n_step_return_and_bootstrap_mask",
    "_tensor_to_scalar",
    "set_seed",
]
