# TMRL config package

Configuration is loaded from **`~/TmrlData/config/config.json`**. This package exposes it as Python constants and builds runtime objects (interface, memory, agent, trainer) from it.

## Where to change what

| What you want to change | Where it lives | In config.json |
|-------------------------|----------------|----------------|
| Run name, buffer sizes, CUDA, gamepad, server IPs | `constants.py`, `paths.py` (top-level / paths / network) | `RUN_NAME`, `BUFFERS_MAXLEN`, `CUDA_*`, `VIRTUAL_GAMEPAD`, `LOCALHOST_*`, `PUBLIC_IP_SERVER` |
| Observation type (Lidar vs images), map, rewards, failure rules, image size | `constants.py` → “Environment” section | `ENV` → `RTGYM_INTERFACE`, `MAP_NAME`, `*_REWARD`, `*_FAILURE`, `IMG_*`, `WINDOW_*` |
| Training loop (epochs, rounds, steps, update intervals, memory size, batch size) | `constants.py` → “Model” section | `MODEL` → `MAX_EPOCHS`, `ROUNDS_PER_EPOCH`, `TRAINING_STEPS_PER_ROUND`, `UPDATE_*`, `MEMORY_SIZE`, `BATCH_SIZE`, `BEST_CHECKPOINT_CRITERION` ("eval" or "train"), `BEST_CHECKPOINT_LAP_TIME`, `BEST_CHECKPOINT_MIN_FINISHES`, `COMPETITION_EVAL_*` |
| Worker eval frequency / runs per eval | top-level `TMRL` | `RW_TEST_EPISODE_INTERVAL`, `RW_TEST_EPISODES_PER_EVAL` (e.g. 10 for competition-style eval) |
| Network architecture (CNN/RNN/MLP sizes, dropout, scheduler) | `constants.py` → “Model architecture” | `MODEL` → `CNN_FILTERS`, `RNN_*`, `API_MLP_SIZES`, `SCHEDULER` |
| Algorithm (SAC/TQC/REDQ), learning rates, gamma, quantiles | `constants.py` → “Algorithm” section | `ALG` → `ALGORITHM`, `LR_*`, `GAMMA`, `POLYAK`, `QUANTILES_*`, etc. |
| Debug / profiling | `constants.py` → “Debug / profiling” | `DEBUGGER` |
| Weights & Biases | `constants.py` → “Weights & Biases” | `WANDB_*` |
| Ports, TLS, buffer size | `constants.py` → “Networking” | `PORT`, `LOCAL_PORT_*`, `PASSWORD`, `TLS_*`, `BUFFER_SIZE`, `HEADER_SIZE` |
| Which interface / memory / agent / trainer are used | `config_objects.py` | Driven by `ENV.RTGYM_INTERFACE` and `ALG.ALGORITHM` |

## Modules

- **`loader.py`**  
  Loads `config.json`, validates version, merges with defaults, and exposes:
  - `TMRL_CONFIG`, `ENV_CONFIG`, `DEBUGGER_CONFIG`
  - `create_config()`: flat dict for training agent / checkpoint loading

- **`constants.py`**  
  Derived from loaded config; exposes:
  - Path constants (e.g. `TMRL_FOLDER`, `CHECKPOINTS_FOLDER`, `MODEL_PATH_WORKER`) via `paths.py`
  - Environment flags and scalars (from `ENV`)
  - Model/algorithm/debug/network constants

- **`paths.py`**  
  Path constants (checkpoints, dataset, reward, track, etc.).

- **`defaults.py`**  
  Default config values for backward compatibility.

- **`models.py`**, **`enums.py`**  
  Pydantic models and enums for config validation.

- **`config_objects.py`**  
  Uses `constants` and `paths` to select and partially apply:
  - **TRAIN_MODEL / POLICY** – neural net classes
  - **RTGYM_INTERFACE_CLASS** – rtgym interface (TM2020* with kwargs)
  - **CONFIG_DICT** – rtgym config
  - **SAMPLE_COMPRESSOR**, **OBS_PREPROCESSOR**
  - **MEM / MEMORY** – replay memory class (partial)
  - **AGENT** – SAC/TQC/REDQ/IQN agent (partial)
  - **TRAINER** – TorchTrainingOffline (partial)
  - **DUMP/LOAD_RUN_INSTANCE_FN**, **UPDATER_FN**

Selection is determined by `RTGYM_INTERFACE` (Lidar vs images, which variant) and `ALGORITHM` (SAC, TQC, REDQSAC, IQN). See the module docstring in `config_objects.py` for the decision flow.
