"""Enumeration types for configuration."""

from enum import StrEnum


class AlgorithmName(StrEnum):
    """Supported RL algorithm identifiers in config."""

    SAC = "SAC"
    TQC = "TQC"
    REDQSAC = "REDQSAC"
    IQN = "IQN"
    SDSAC = "SDSAC"
