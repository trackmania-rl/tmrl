# Changelog

## [0.8.0] - 2026-03-21

### Added
- New algorithm support and configuration hooks for IQN, TQC, REDQ-SAC, and SDSAC-style workflows.
- New TrackMania interface and observation-space variants, including helpers for TQC-oriented observation definitions.
- New replay-memory and sequence-oriented handling paths to support broader training setups (including recurrent pipelines).
- New utility scripts for reward trajectory generation, interpolation, plotting, and player-run alignment workflows.
- Expanded automated tests for replay logic, interface observation contracts, numeric training behavior, and integration edge cases.
- Developer-focused tooling improvements (`Makefile` targets and helper scripts) for more reproducible local validation.

### Changed
- Packaging metadata was modernized around `pyproject.toml`, with release version aligned to `0.8.0`.
- Configuration system was heavily refactored (`defaults`, `constants`, `loader`, typed models) for better consistency and maintainability.
- Offline training orchestration was reworked to better integrate algorithm options, replay flow, and diagnostics.
- Model and actor modules (MLP/CNN/RNN/IMPALA/Sophy families) were harmonized for more consistent construction and optimizer behavior.
- Networking and memory flow were refined to improve coordination between workers, server, and trainer components.
- Runtime tools (`record*`, environment checks, init helpers) were updated for improved reliability and compatibility.

### Fixed
- Import/startup edge cases in package initialization and environment bootstrap paths.
- Replay-pipeline consistency issues in sample handling and debug/validation paths.
- Stability and correctness edge cases in training utilities, validated by stronger test coverage.
- Compatibility mismatches between some interfaces, observation spaces, and model expectations.

### Removed
- Legacy monolithic modules superseded by package-structured implementations:
  - `tmrl/config/config_constants.py`
  - `tmrl/custom/custom_algorithms.py`
  - `tmrl/custom/custom_models.py`


### Notes
- This release touches most major subsystems (`config`, `algorithms`, `interfaces`,
  `models`, `memory/networking`, `tools`, and `tests`).
