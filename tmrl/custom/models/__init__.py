"""TrackMania custom models: MLP, residual MLP, CNN, EfficientNet-based actor-critics.

This package contains various neural network architectures for reinforcement
learning in TrackMania, including:

- MLP-based actor-critics (standard SAC)
- Residual MLP actor-critics (improved gradient flow)
- EfficientNet-based models (frozen/trainable image encoders)
- Vanilla CNN models (lightweight image processing)
- RNN models (sequential decision making)
- GNN+EffNet+Sophy hybrid models (state-of-the-art architecture)
"""

# Base utilities
from tmrl.custom.models.base import (
    EPSILON,
    LOG_STD_MAX,
    LOG_STD_MIN,
    EffNetV2,
    MBConv,
    SELayer,
    SiLU,
    _cat_obs,
    _cat_obs_except_image,
    _ensure_float,
    _make_divisible,
    _obs_dim,
    _obs_spaces_list,
    _vector_dim_except,
    combined_shape,
    conv2d_out_dims,
    conv_1x1_bn,
    conv_3x3_bn,
    count_vars,
    effnetv2_l,
    effnetv2_m,
    effnetv2_s,
    effnetv2_xl,
    effnetv2_xs,
    mlp,
    num_flat_features,
)

# CNN models
from tmrl.custom.models.cnn import (
    SquashedGaussianVanillaCNNActor,
    SquashedGaussianVanillaColorCNNActor,
    VanillaCNN,
    VanillaCNNActorCritic,
    VanillaCNNQFunction,
    VanillaColorCNNActorCritic,
    VanillaColorCNNQFunction,
    remove_colors,
)

# EfficientNet models
from tmrl.custom.models.efficientnet import (
    EffNetActorCritic,
    EffNetQFunction,
    FrozenEffNetResidualActorCritic,
    FrozenEffNetResidualQFunction,
    SquashedGaussianEffNetActor,
    SquashedGaussianFrozenEffNetResidualActor,
)

# GNN+EffNet+Sophy hybrid models
from tmrl.custom.models.gnn_effnet_sophy import (
    GnnEffNetSophyResidualActorCritic,
    QRCNNGnnEffNetSophyResidual,
    SquashedActorGnnEffNetSophyResidual,
    _build_track_gnn_branch,
    _obs_to_flat_tensor,
    _TrackGNN,
)

# MLP models
from tmrl.custom.models.mlp import (
    MLPActorCritic,
    MLPQFunction,
    REDQMLPActorCritic,
    SquashedGaussianMLPActor,
)

# Residual MLP models
from tmrl.custom.models.residual import (
    REDQResidualMLPActorCritic,
    ResidualMLPActorCritic,
    ResidualMLPQFunction,
    SquashedGaussianResidualMLPActor,
)

# RNN models
from tmrl.custom.models.rnn import (
    RNNActorCritic,
    RNNQFunction,
    SquashedGaussianRNNActor,
    rnn,
)

__all__ = [
    "EPSILON",
    # Constants
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "EffNetActorCritic",
    "EffNetQFunction",
    "EffNetV2",
    "FrozenEffNetResidualActorCritic",
    "FrozenEffNetResidualQFunction",
    "GnnEffNetSophyResidualActorCritic",
    "MBConv",
    "MLPActorCritic",
    "MLPQFunction",
    "QRCNNGnnEffNetSophyResidual",
    "REDQMLPActorCritic",
    "REDQResidualMLPActorCritic",
    "RNNActorCritic",
    "RNNQFunction",
    "ResidualMLPActorCritic",
    "ResidualMLPQFunction",
    "SELayer",
    # Base utilities
    "SiLU",
    "SquashedActorGnnEffNetSophyResidual",
    "SquashedGaussianEffNetActor",
    # EfficientNet
    "SquashedGaussianFrozenEffNetResidualActor",
    # MLP
    "SquashedGaussianMLPActor",
    "SquashedGaussianRNNActor",
    # Residual MLP
    "SquashedGaussianResidualMLPActor",
    "SquashedGaussianVanillaCNNActor",
    "SquashedGaussianVanillaColorCNNActor",
    # CNN
    "VanillaCNN",
    "VanillaCNNActorCritic",
    "VanillaCNNQFunction",
    "VanillaColorCNNActorCritic",
    "VanillaColorCNNQFunction",
    # GNN+EffNet+Sophy
    "_TrackGNN",
    "_build_track_gnn_branch",
    "_cat_obs",
    "_cat_obs_except_image",
    "_ensure_float",
    "_make_divisible",
    "_obs_dim",
    "_obs_spaces_list",
    "_obs_to_flat_tensor",
    "_vector_dim_except",
    "combined_shape",
    "conv2d_out_dims",
    "conv_1x1_bn",
    "conv_3x3_bn",
    "count_vars",
    "effnetv2_l",
    "effnetv2_m",
    "effnetv2_s",
    "effnetv2_xl",
    "effnetv2_xs",
    "mlp",
    "num_flat_features",
    "remove_colors",
    # RNN
    "rnn",
]
