# TMRL quick reference guide

## Quick links
- [config.json file](#configuration-file)
- [Command line interface](#command-line-interface)
- [Documentation (Python library)](https://tmrl.readthedocs.io/en/latest/)

## Configuration file

The `config.json` file contains the default parameters used by the `tmrl` library.
It is located under `~/TmrlData/config`.

Customizing the example `tmrl` pipeline for TrackMania can be achieved by simply modifying this file.

In case you break your `config.json`, you can always find valid examples under `~/TmrlData/resources` to replace the content of your `config.json` file with a text editor.

All parameters are described below.

:warning: `tmrl` does not support comments in `config.json`, we provide them here only for documentation: do not copy-paste those comments!

```json5
{
  "RUN_NAME": "SAC_4_imgs_pretrained",  // experiment name (matches names in checkpoints and weights)
  "RESET_TRAINING": false,  // if true, training restarts from scratch (loads the replay buffer only)
  "BUFFERS_MAXLEN": 500000,  // maximum length of local buffers (this is NOT the replay buffer)
  "RW_MAX_SAMPLES_PER_EPISODE": 1000,  // tmrl forces truncation if the episode is longer than this
  "CUDA_TRAINING": true,  //  if true, training happens on GPU (Trainer)
  "CUDA_INFERENCE": false,  // if true, inference happens on GPU (RolloutWorker)
  "VIRTUAL_GAMEPAD": true,  // if true, the example TrackMania pipeline uses vgamepad
  "LOCALHOST_WORKER": true,  // must be false when the Server is not on localhost
  "LOCALHOST_TRAINER": true,  // must be false when the Server is not on localhost
  "PUBLIC_IP_SERVER": "0.0.0.0",  // Server IP when not on localhost
  "PASSWORD": "YourRandomPasswordHere",  // needs to match on all machines (read the Security section)
  "TLS": false,  // IMPORTANT: true when using tmrl on a public network (read the Security section)
  "TLS_HOSTNAME": "default",  // TLS hostname (for custom tlspyo configuration only)
  "TLS_CREDENTIALS_DIRECTORY": "",  // TLS credential directory (for custom tlspyo configuration only)
  "NB_WORKERS": -1,  // maximum number of Workers that can connect to the Server (-1 for infinite)
  "WANDB_PROJECT": "tmrl",  // your wandb project name
  "WANDB_ENTITY": "tmrl",  // your wandb entity name
  "WANDB_KEY": "YourWandbApiKey",  // your wandb key
  "PORT": 55555,  // public port of your Server
  "LOCAL_PORT_SERVER": 55556,  // localhost Server port (must not overlap)
  "LOCAL_PORT_TRAINER": 55557,  // localhost Trainer port (must not overlap)
  "LOCAL_PORT_WORKER": 55558,  // CAUTION: change this manually if several workers are on the same machine!
  "BUFFER_SIZE": 536870912,  // tslpyo buffer size (NOT the replay buffer)
  "HEADER_SIZE": 12,  // tlspyo header size (bytes)
  "MAX_EPOCHS": 10000,  // maximum number of training "epochs" (checkpoint and wandb after each "epoch")
  "ROUNDS_PER_EPOCH": 100,  // "rounds" per training "epoch" (metrics displayed after each "round")
  "TRAINING_STEPS_PER_ROUND": 200,  // number of training iterations per "round"
  "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP": 4.0,  // pause training if not enough samples
  "ENVIRONMENT_STEPS_BEFORE_TRAINING": 1000,  // minimum number of samples before training starts
  "UPDATE_MODEL_INTERVAL": 200,  // the model is updated at this interval of training steps
  "UPDATE_BUFFER_INTERVAL": 200,  // samples are retrieved at this interval of training steps
  "SAVE_MODEL_EVERY": 0,  // save a copy of the model at this interval of model updates (0: no history)
  "MEMORY_SIZE": 1000000,  // REPLAY BUFFER SIZE (maximum number of samples in the Memory)
  "BATCH_SIZE": 256,  // training batch size
  "ALG": {  // this section contains parameters of the training algorithm:
    "ALGORITHM": "SAC",  // algorithm name ("SAC", "REDQSAC")
    "LEARN_ENTROPY_COEF":false,  // true for SACv2
    "LR_ACTOR":0.00001,  // learning rate of the actor
    "LR_CRITIC":0.00005,  // learning rate of the critic
    "LR_ENTROPY":0.0003,  // learning rate of the entropy factor (SACv2)
    "GAMMA":0.995,  // discount factor
    "POLYAK":0.995,  // polyak averaging factor of the target critic
    "TARGET_ENTROPY":-0.5,  // entropy (SACv2)
    "ALPHA":0.01,  // entropy factor (constant for SAC, initial for SACv2)
    "REDQ_N":10,  // number of critic networks (REDQSAC)
    "REDQ_M":2,  // random subset size (REDQSAC)
    "REDQ_Q_UPDATES_PER_POLICY_UPDATE":20,  // (for REDQSAC)
    "OPTIMIZER_ACTOR": "adam",  // actor optimizer ("adam", "adamw", "sgd")
    "OPTIMIZER_CRITIC": "adam",  // critic optimizer ("adam", "adamw", "sgd")
    "BETAS_ACTOR": [0.997, 0.997],  // actor betas (for Adam and AdamW)
    "BETAS_CRITIC": [0.997, 0.997],  // critic betas (for Adam and AdamW)
    "L2_ACTOR": 0.0,  // actor weight decay (for Adam and AdamW)
    "L2_CRITIC": 0.0  // critic weight decay (for Adam and AdamW)
  },
  "ENV": {  // this section contains environment parameters:
    "RTGYM_INTERFACE": "TM20FULL",  // environment name ("TM20FULL", "TM20LIDAR")
    "WINDOW_WIDTH": 256,  // the TM window is resized to the width
    "WINDOW_HEIGHT": 128,  // the TM window is resized to the height
    "IMG_WIDTH": 64,  // the screenshots fed to the model are resized to this width
    "IMG_HEIGHT": 64,  // the screenshots fed to the model are resized to this height
    "IMG_GRAYSCALE": true,  // if true, images are converted to grayscale
    "SLEEP_TIME_AT_RESET": 1.5,  // to wait for the green light after respawn
    "IMG_HIST_LEN": 4,  // length of the screenshot/lidar history fed to the model
    "RTGYM_CONFIG": {  // this section contains rtgym-specific parameters
      "time_step_duration": 0.05,  // real-time target time-step duration (s)
      "start_obs_capture": 0.04,  // observation capture starts after this duration (s)
      "time_step_timeout_factor": 1.0,  // maximum allowed elasticity for time-step duration
      "act_buf_len": 2,  // number of actions in the action history fed to the model
      "benchmark": false,  // set this to true when using the --benchmark command
      "wait_on_done": true,  // true in the example TrackMania pipeline
      "ep_max_length": 1000,  // rtgym truncates episodes after this number of time-steps
      "interface_kwargs": {"save_replays": false}  // true saves replays in TrackMania
    },
    "REWARD_CONFIG": {
      "END_OF_TRACK": 100.0,  // the agent gets this when crossing the finish line
      "CONSTANT_PENALTY": 0.0,  // constant reward
      "CHECK_FORWARD": 500,  // longer is more computational but enables longer cuts
      "CHECK_BACKWARD": 10,  // longer is more computational but more reliable (10 is fine)
      "FAILURE_COUNTDOWN": 10,  // no reward for X consecutive time-steps = episode terminated
      "MIN_STEPS": 70,  // initial number of time-steps before episode can get terminated
      "MAX_STRAY": 100.0  // if the car wanders further away, the episode is terminated (m)
    }
  },
  "__VERSION__": "0.6.0"  // (compatibility check, do not modify)
}
```


## Command line interface

### General:

- Regenerate the `TmrlData` folder to its default state after it has been deleted, and/or display its location on your machine:
    ```bash
    python -m tmrl --install
    ```

- Launch a test worker (does not collect training samples):
    ```bash
    python -m tmrl --test
    ```

- Record a reward function in TrackMania. You should start recording at the beginning of the track. The script will compute the reward function automatically when you cross the finish line:
    ```bash
    python -m tmrl --record-reward
    ```

- Check the sanity of your reward function and camera setting when customizing the `Full` or `Lidar` environments in TrackMania. When using the `Full` environment, the history of images used by the model will be displayed. When using the `Lidar` environment, the rangefinder will be displayed instead. When using `Lidar`, check that the blue lines stop at sensible positions (they start in the bottom-center and stop when finding black pixels). You also want to drive around and check that the rewards printed in the console make sense:
    ```bash
    python -m tmrl --check-environment
    ```

- Launch the `tmrl` server:
    ```bash
    python -m tmrl --server
    ```

- Launch the `tmrl` trainer:
    ```bash
    python -m tmrl --trainer
    ```

- Add this option to your `--trainer` command if you wish your training metrics to be logged on `wandb`. By default, metrics are logged on a public project that we clear once in a while. To use your own `wandb` account instead, edit the relevant entries of `config.json`:
    ```bash
    --wandb
    ```

- Launch a `tmrl` worker:
    ```bash
    python -m tmrl --worker
    ```

### Advanced:

- Launch a `tmrl` worker that ignores model updates. This is useful for collecting off-policy samples from an expert model:
    ```bash
    python -m tmrl --expert
    ```

- Benchmark your environment performance. This requires the `"benchmark"` entry to be set to `true` in `config.json`:
    ```bash
    python -m tmrl --benchmark
    ```

- Add this option to your `tmrl` commands to modify the content of `config.json` at runtime. The entries that you wish to modify need to be written in Python format (e.g., `{"CUDA_TRAINING": True, "ALG": {"ALGORITHM": "REDQSAC"}}`):
    ```bash
    --config={...}
    ```

## API reference:
Read the [TMRL documentation](https://tmrl.readthedocs.io/en/latest/).
