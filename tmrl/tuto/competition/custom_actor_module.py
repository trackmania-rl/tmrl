"""
==============================================================================
COMPETITION TUTORIAL #1: Custom model and RL algorithm
==============================================================================

In this tutorial, we will customize the TrackMania training pipeline.

The tutorial works with the TrackMania FULL Gymnasium environment.
Please refer to the README on GitHub to set up this environment in config.json:
https://github.com/trackmania-rl/tmrl#full-environment

Note: This tutorial describes implementing and running a TrainingAgent along with an ActorModule.
It is relevant if you want to implement your own RL approaches in TrackMania.
If you plan to try non-RL approaches instead, this is also accepted in the competition:
just use the Gymnasium Full environment and do whatever you need,
then, wrap your trained policy in an ActorModule, and submit your entry :)

Copy and adapt this script to implement your own algorithm/model in TrackMania.
Then, use the script as follows:

To launch the Server, provided the script is named custom_actor_module.py, execute:
python custom_actor_module.py --server

In another terminal, launch the Trainer:
python custom_actor_module.py --trainer

And in yet another terminal, launch a RolloutWorker:
python custom_actor_module.py --worker

You can launch these in any order, but we recommend server, then trainer, then worker.
If you are running everything on the same machine, your trainer may consume all your resource,
resulting in your worker struggling to collect samples in a timely fashion.
If your worker crazily warns you about time-steps timing out, this is probably the issue.
The best way is to run worker(s) and trainer on separate machines.
The server can run on either machine or a third one both can reach.
Achieving this is easy (and is the whole point of the TMRL framework).
Just adapt config.json (or this script) to your network configuration.
In particular, you will want to set the following in the TMRL config.json file of all your machines:

"LOCALHOST_WORKER": false,
"LOCALHOST_TRAINER": false,
"PUBLIC_IP_SERVER": "<ip.of.the.server>",
"PORT": <port of the server (usually requires port forwarding if accessed via the Internet)>,

If you are training over the Internet, please read the security instructions on the
TMRL GitHub page.

IMPORTANT: Set a custom 'RUN_NAME' in config.json, otherwise this script will not work.
"""

# Let us start our tutorial by importing some useful stuff.

# The constants that are defined in config.json:
import itertools
import json
import os
from copy import deepcopy
from math import floor

# And a couple external libraries:
import numpy as np
import tmrl.config as cfg

# Useful classes:
import tmrl.config.config_objects as cfg_obj
import torch
import torch.nn as nn
import torch.nn.functional as F
from tmrl.actor import TorchActorModule
from tmrl.custom.utils.nn import copy_shared, no_grad

# The TMRL three main entities (i.e., the Trainer, the RolloutWorker and the central Server):
from tmrl.networking import RolloutWorker, Server, Trainer
from tmrl.training import TrainingAgent

# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import cached_property, partial
from torch.distributions.normal import Normal
from torch.optim import Adam

# The training class that we will customize with our own training algorithm in this tutorial:
from training_offline import TrainingOffline

# Now, let us look into the content of config.json:

# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# You can change these parameters here directly (not recommended),
# or you can change them in the TMRL config.json file (recommended).

# Maximum number of training 'epochs':
# (training is checkpointed at the end of each 'epoch',
# this is also when training metrics can be logged to wandb).
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# Number of rounds per 'epoch':
# (training metrics are displayed in the terminal at the end of each round)
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# Number of training steps per round:
# (a training step is a call to the train() function that we will define later in this tutorial)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# Minimum number of environment steps collected before training starts:
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# Maximum training steps / environment steps ratio:
# (if training becomes faster than this ratio, it will be paused,
# waiting for new samples from the environment).
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# Number of training steps performed between broadcasts of policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# Training steps between retrievals of received samples into replay buffer:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Training device (e.g., "cuda:0"):
device_trainer = "cuda" if cfg.CUDA_TRAINING else "cpu"

# Maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# Batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

# Wandb credentials:
# (Change this with your own if you want to keep your training curves private)
# (Also, please use your own wandb account if you are going to log huge stuff :) )

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
wandb_project = cfg.TMRL_CONFIG[
    "WANDB_PROJECT"
]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ["WANDB_API_KEY"] = wandb_key  # this line sets your wandb API key as the active key

# Number of time-steps after which episodes collected by the worker are truncated:
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# Networking parameters:
# (In TMRL, networking is managed by tlspyo. The following are tlspyo parameters.)
server_ip_for_trainer = (
    cfg.SERVER_IP_FOR_TRAINER
)  # IP of the machine running the Server (trainer point of view)
server_ip_for_worker = (
    cfg.SERVER_IP_FOR_WORKER
)  # IP of the machine running the Server (worker point of view)
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
security = cfg.SECURITY  # when training over the Internet, it is safer to change this to "TLS"
# (please read the security instructions on GitHub)


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================
# You may want to change the following in advanced applications;
# however, most competitors will not need to change this.
# If interested, read the full TMRL tutorial on GitHub.
# These parameters are to change here directly (if you want).
# (Note: The tutorial may stop working if you change these)

# Base class of the replay memory used by the trainer:
memory_base_cls = cfg_obj.MEM

# Sample compression scheme applied by the worker for this replay memory:
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR

# Sample preprocessor for data augmentation:
sample_preprocessor = None

# Path from where an offline dataset can be loaded to initialize the replay memory:
dataset_path = cfg.DATASET_PATH

# Preprocessor applied by the worker to the observations it collects:
# (Note: if your script defines the name "obs_preprocessor",
# we will use your preprocessor instead of the default).
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================
# Competitors CANNOT change the following parameters.

# rtgym environment class (full TrackMania Gymnasium environment):
env_cls = cfg_obj.ENV_CLS

# Device used for inference on workers
# (change if you like but keep in mind that the competition evaluation is on CPU).
device_worker = "cpu"

# =====================================================================
# ENVIRONMENT PARAMETERS
# =====================================================================
# You are allowed to customize these environment parameters.
# Do not change these here though, customize them in config.json.
# Your environment configuration must be part of your submission,
# e.g., the "ENV" entry of your config.json file.

# Dimensions of the TrackMania window:
window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488

# Dimensions of the actual images in observations:
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT

# Whether you are using grayscale (default) or color images:
# (Note: The tutorial will stop working if you use colors)
img_grayscale = cfg.GRAYSCALE

# Number of consecutive screenshots in each observation:
imgs_buf_len = cfg.IMG_HIST_LEN

# Number of actions in the action buffer (this is part of observations):
# (Note: The tutorial will stop working if you change this)
act_buf_len = cfg.ACT_BUF_LEN

# =====================================================================
# MEMORY CLASS
# =====================================================================
# Nothing to do here.
# This is the memory class passed to the Trainer.
# If you need a custom memory, change the relevant advanced parameters.
# Custom memories are described in the full TMRL tutorial.

memory_cls = partial(
    memory_base_cls,
    memory_size=memory_size,
    batch_size=batch_size,
    sample_preprocessor=sample_preprocessor,
    dataset_path=cfg.DATASET_PATH,
    imgs_obs=imgs_buf_len,
    act_buf_len=act_buf_len,
    crc_debug=False,
)

# =====================================================================
# CUSTOM MODEL
# =====================================================================
# Alright, now for the fun part.
# Our goal in this competition is to come up with the best trained
# ActorModule for TrackMania 2020, where an 'ActorModule' is a policy.
# In this tutorial, we present a deep RL way of tackling this problem:
# we implement our own deep neural network architecture (ActorModule),
# and then we implement our own RL algorithm to train this module.

# We will implement SAC and a hybrid CNN/MLP model.

# The following constants are from the Spinup implementation of SAC
# that we simply copy/paste and adapt in this tutorial.
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# Let us use the ActorModule we are supposed to implement (imports are at top).
# In the full version of the TrackMania 2020 environment, the
# observation-space comprises a history of screenshots. Thus, we need
# Computer Vision layers such as a CNN in our model to process these.
# The observation space also comprises single floats representing speed,
# rpm and gear. We will merge these with the information contained in
# screenshots thanks to an MLP following our CNN layers.


# Here is the MLP:
def mlp(sizes, activation, output_activation=nn.Identity):
    """
    A simple MLP (MultiLayer Perceptron).

    Args:
        sizes: list of integers representing the hidden size of each layer
        activation: activation function of hidden layers
        output_activation: activation function of the last layer

    Returns:
        Our MLP in the form of a Pytorch Sequential module
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# The next utility computes the dimensionality of CNN feature maps when flattened together:
def num_flat_features(x):
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# The next utility computes the dimensionality of the output in a 2D CNN layer:
def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor(
        (
            h_in
            + 2 * conv_layer.padding[0]
            - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1)
            - 1
        )
        / conv_layer.stride[0]
        + 1
    )
    w_out = floor(
        (
            w_in
            + 2 * conv_layer.padding[1]
            - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1)
            - 1
        )
        / conv_layer.stride[1]
        + 1
    )
    return h_out, w_out


# Let us now define a module that will be the main building block of both our actor and critic:
class VanillaCNN(nn.Module):
    def __init__(self, q_net):
        """
        Simple CNN (Convolutional Neural Network) model for SAC (Soft Actor-Critic).

        Args:
            q_net (bool): indicates whether this neural net is a critic network
        """
        super().__init__()

        self.q_net = q_net

        # Convolutional layers processing screenshots:
        # The default config.json gives 4 grayscale images of 64 x 64 pixels
        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        # Dimensionality of the CNN output:
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Dimensionality of the MLP input:
        # The MLP input will be formed of:
        # - the flattened CNN output
        # - the current speed, gear and RPM measurements (3 floats)
        # - the 2 previous actions (2 x 3 floats),
        # important because of the real-time nature of our controller
        # - when the module is the critic, the selected action (3 floats)
        float_features = 12 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        # MLP layers: policy samples from a gaussian defined later; critic output dim 1.
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features, *self.mlp_layers], nn.ReLU)

    def forward(self, x):
        """
        Forward pass: neural network computes output from input.

        Args:
            x (torch.Tensor): input tensor (i.e., the observation fed to our deep neural network)

        Returns:
            the output of our neural network in the form of a torch.Tensor
        """
        if self.q_net:
            # The critic takes the next action (act) as additional input
            # act1 and act2 are the actions in the action buffer (real-time RL):
            speed, gear, rpm, images, act1, act2, act = x
        else:
            # For the policy, the next action (act) is what we are computing, so we don't have it:
            speed, gear, rpm, images, act1, act2 = x

        # Forward pass of our images in the CNN:
        # (note that the competition environment outputs histories of 4 images
        # and the default config outputs these as 64 x 64 greyscale,
        # we will stack these greyscale images along the channel dimension of our input tensor)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Now we will flatten our output feature map.
        # Let us double-check that our dimensions are what we expect them to be:
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, (
            f"x.shape:{x.shape},\
                                                    flat_features:{flat_features},\
                                                    self.out_channels:{self.out_channels},\
                                                    self.h_out:{self.h_out},\
                                                    self.w_out:{self.w_out}"
        )
        # All good, let us flatten our output feature map:
        x = x.view(-1, flat_features)

        # Finally, we can feed the result along our float values to the MLP:
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network :)
        return x


# We implement the TMRL ActorModule interface to submit for this competition.

# During training, TMRL saves our ActorModule in TmrlData/weights.
# Default is torch (pickle); competition does not accept pickle (security).
# We submit weights as human-readable JSON; we override save()/load() with JSON.


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)  # noqa: B026

    def object_hook(self, dct):
        for key in dct:
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


class MyActorModule(TorchActorModule):
    """
    Our policy wrapped in the TMRL ActorModule class.

    The only required method is ActorModule.act().
    We also implement a forward() method for our training algorithm.

    (Note: TorchActorModule is a subclass of ActorModule and torch.nn.Module)
    """

    def __init__(self, observation_space, action_space):
        """
        __init__ takes observation_space and action_space.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
        """
        # We must call the superclass __init__:
        super().__init__(observation_space, action_space)

        # And initialize our attributes:
        dim_act = action_space.shape[0]  # dimensionality of actions
        act_limit = action_space.high[0]  # maximum amplitude of actions
        # Our hybrid CNN+MLP policy:
        self.net = VanillaCNN(q_net=False)
        # Policy output: sample from gaussian (means...)
        self.mu_layer = nn.Linear(256, dim_act)
        # ... and log standard deviations:
        self.log_std_layer = nn.Linear(256, dim_act)
        # We will squash this within the action space thanks to a tanh final activation:
        self.act_limit = act_limit

    def save(self, path):
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        IMPORTANT: FOR THE COMPETITION, WE ONLY ACCEPT JSON AND PYTHON FILES.
        IN PARTICULAR, WE *DO NOT* ACCEPT PICKLE FILES (such as output by torch.save()...).

        All your submitted files must be human-readable, for everyone's safety.
        Indeed, untrusted pickle files are an open door for hackers.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, "w") as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        # torch.save(self.state_dict(), path)

    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Adapt this method to your submission so that we can load your trained ActorModule.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        self.device = device
        with open(path) as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        # self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def forward(self, obs, test=False, compute_logprob=True):
        """
        Computes the policy action from the observation.

        Deep RL trains the actor to output good actions; the critic is for training only.
        Our ActorModule implements only the actor.

        Args:
            obs: Gymnasium observation (TorchActorModule: torch.Tensor)
            test (bool): True at test (deployment), False when training;
                SAC samples randomly in training, deterministically at test.
            compute_logprob (bool): SAC sets True to get log probabilities.

        Returns:
            the action sampled from our policy from observation obs
            the log probability of this action (this will be used for SAC)
        """
        # obs is our input observation.
        # We feed it to our actor neural network, which will output an action.

        # Let us feed it to our MLP:
        net_out = self.net(obs)
        # Now, the means of our multivariate gaussian (i.e., Normal law) are:
        mu = self.mu_layer(net_out)
        # And the corresponding standard deviations are:
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # We can now sample our action in the resulting multivariate gaussian (Normal) distribution:
        pi_distribution = Normal(mu, std)
        # at test time, our action is deterministic (it is just the means)
        pi_action = mu if test else pi_distribution.rsample()
        # Log probs of the gaussian, for SAC:
        if compute_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # (next line: TanH squashing correction from Spinup SAC)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None
        # And we squash our action within the action space:
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        # Finally, we remove the batch dimension:
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    # Now, the only method that all participants are required to implement is act()
    # act() is the interface for TMRL to use your ActorModule as the policy it tests in TrackMania.
    # For the evaluation, the "test" argument will be set to True.
    def act(self, obs, test=False):
        """
        Computes an action from an observation.

        This method is the one all participants must implement.
        It is the policy that TMRL will use in TrackMania to evaluate your submission.

        Args:
            obs (object): input observation (TorchActorModule: torch.Tensor)
            test (bool): True at test-time (e.g. evaluation), False otherwise

        Returns:
            act (numpy.array): action as numpy array of 3 values in [-1.0, 1.0]
        """
        # Policy is a neural network; act() is straightforward.
        # Log probs are not needed here (used in SAC training).
        # TorchActorModule: TMRL calls act() in torch.no_grad() context.
        # We still use no_grad() for users of ActorModule instead of TorchActorModule.
        with torch.no_grad():
            a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
            return a.cpu().numpy()


# The critic module for SAC is now super straightforward:
class VanillaCNNQFunction(nn.Module):
    """
    Critic module for SAC.
    """

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)  # q_net is True for a critic module

    def forward(self, obs, act):
        """
        Estimates the action-value of the (obs, act) state-action pair.

        In RL theory, the action-value is the expected sum of (gamma-discounted) future rewards
        when observing obs, taking action act, and following the current policy ever after.

        Args:
            obs: current observation
            act: tried next action

        Returns:
            The action-value of act in situation obs, as estimated by our critic network
        """
        # q_net is True: append action act to observation obs.
        # obs is a tuple of batched tensors: history of 4 images, speed, etc.
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)


# Finally, let us merge this together into an actor-critic torch.nn.module for training.
# Classically, we use one actor and two parallel critics to alleviate the overestimation bias.
class VanillaCNNActorCritic(nn.Module):
    """
    Actor-critic module for the SAC algorithm.
    """

    def __init__(self, observation_space, action_space):
        super().__init__()

        # Policy network (actor):
        self.actor = MyActorModule(observation_space, action_space)
        # Value networks (critics):
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)


# =====================================================================
# CUSTOM TRAINING ALGORITHM
# =====================================================================
# So far, we have implemented our custom model.
# We have also wrapped it in an ActorModule, which we will train and
# submit as an entry to the TMRL competition.
# Our ActorModule will be used in Workers to collect training data.
# Our VanillaCNNActorCritic will be used in the Trainer for training
# this ActorModule. Let us now tackle the training algorithm per-se.
# In TMRL, this is done by implementing a custom TrainingAgent.

# A TrainingAgent must implement two methods:
# -> train(batch): optimizes the model from a batch of RL samples
# -> get_actor(): outputs a copy of the current ActorModule
# In this tutorial, we implement the Soft Actor-Critic algorithm
# by adapting the OpenAI Spinup implementation.


class SACTrainingAgent(TrainingAgent):
    """
    Our custom training algorithm (SAC in this tutorial).

    Custom TrainingAgents implement two methods: train(batch) and get_actor().
    The train method performs a training step.
    The get_actor method retrieves your ActorModule to save it and send it to the RolloutWorkers.

    Your implementation must also pass three required arguments to the superclass:

    - observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
    - action_space (gymnasium.spaces.Space): action space (here for your convenience)
    - device (str): device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)
    """

    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(
        self,
        observation_space=None,  # Gymnasium observation space (convenience)
        action_space=None,  # Gymnasium action space (convenience)
        device=None,  # Device for training (required)
        model_cls=VanillaCNNActorCritic,  # An actor-critic module, encapsulating our ActorModule
        gamma=0.99,  # Discount factor
        polyak=0.995,  # Exponential averaging factor for the target critic
        alpha=0.2,  # Value of the entropy coefficient
        lr_actor=1e-3,  # Learning rate for the actor
        lr_critic=1e-3,
    ):  # Learning rate for the critic

        # required arguments passed to the superclass:
        super().__init__(
            observation_space=observation_space, action_space=action_space, device=device
        )

        # custom stuff:
        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        """
        Returns a copy of the current ActorModule.

        We return a copy without gradients, as this is for sending to the RolloutWorkers.

        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor

    def train(self, batch):
        """
        Executes a training iteration from batched training samples (batches of RL transitions).

        A training sample is of the form (o, a, r, o2, d, t) where:
        -> o is the initial observation of the transition
        -> a is the selected action during the transition
        -> r is the reward of the transition
        -> o2 is the final observation of the transition
        -> d is the "terminated" signal indicating whether o2 is a terminal state
        -> t is the "truncated" signal (episode truncated by time-limit)

        We ignore t so that when truncated by time limit, the model does not treat o2 as
        terminal; we treat the episode as if it would continue. With discount factor
        this stays finite. Here the discount factor incentivizes the AI to run fast!

        Args:
            batch: (obs, action, reward, new_obs, terminated, truncated)

        Returns:
            logs: dict of training metrics to log on wandb
        """
        # Decompose batch, ignoring the "truncated" signal:
        o, a, r, o2, d, _ = batch

        # We sample an action in the current policy and retrieve its corresponding log probability:
        pi, logp_pi = self.model.actor(obs=o, test=False, compute_logprob=True)

        # We also compute our action-value estimates for the current transition:
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Now we compute our value target, for which we need to detach from gradients computation:
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha_t * logp_a2)

        # This gives us our critic loss, as the difference between the target and the estimate:
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Optimize critics (step in opposite direction of loss gradient):
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Policy step: detach critics from the gradient graph
        for p in self.q_params:
            p.requires_grad = False

        # Critics' value estimate for the action sampled from current policy:
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Policy loss: minus value estimate plus entropy term
        loss_pi = (self.alpha_t * logp_pi - q_pi).mean()

        # Now we can train our policy in the opposite direction of this loss' gradient:
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # We attach the critics back into the gradient computation graph:
        for p in self.q_params:
            p.requires_grad = True

        # Finally, we update our target model with a slowly moving exponential average:
        with torch.no_grad():
            for p, p_targ in zip(
                self.model.parameters(), self.model_target.parameters(), strict=True
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # TMRL enables us to log training metrics to wandb:
        ret_dict = {
            "loss_actor": loss_pi.detach().item(),
            "loss_critic": loss_q.detach().item(),
        }
        return ret_dict


# Great! We are almost done.
# Now that our TrainingAgent class is defined, let us partially instantiate it.
# SAC has a few hyperparameters that we will need to tune if we want it to work as expected.
# The following have shown reasonable results in the past, using the full TrackMania environment.
# Note however that training a policy with SAC in this environment is a matter of several days!

training_agent_cls = partial(
    SACTrainingAgent,
    model_cls=VanillaCNNActorCritic,
    gamma=0.995,
    polyak=0.995,
    alpha=0.01,
    lr_actor=0.00001,
    lr_critic=0.00005,
)

# =====================================================================
# TMRL TRAINER
# =====================================================================

training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer,
    python_profiling=cfg.PROFILE_TRAINER,
    pytorch_profiling=cfg.PYTORCH_PROFILER,
)

# =====================================================================
# RUN YOUR TRAINING PIPELINE
# =====================================================================
# The training pipeline configured in this tutorial runs with the "TM20FULL" environment.

# You can configure the "TM20FULL" environment by following the instruction on GitHub:
# https://github.com/trackmania-rl/tmrl#full-environment

# In TMRL, a training pipeline has:
# - one Trainer (our training algorithm)
# - one or more RolloutWorker(s) (ActorModule + Gymnasium env)
# - one Server (RolloutWorker(s) and Trainer communicate through it)
# Instantiate via script argument:

if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class CompetitionCli:
        server: bool = False
        trainer: bool = False
        worker: bool = False
        test: bool = False

    args = tyro.cli(CompetitionCli)

    if args.trainer:
        my_trainer = Trainer(
            training_cls=training_cls,
            server_ip=server_ip_for_trainer,
            server_port=server_port,
            password=password,
            security=security,
        )
        my_trainer.run()

        # Note: if you want to log training metrics to wandb, replace my_trainer.run() with:
        # my_trainer.run_with_wandb(entity=wandb_entity,
        #                           project=wandb_project,
        #                           run_id=wandb_run_id)

    elif args.worker or args.test:
        rw = RolloutWorker(
            env_cls=env_cls,
            actor_module_cls=MyActorModule,
            sample_compressor=sample_compressor,
            device=device_worker,
            server_ip=server_ip_for_worker,
            server_port=server_port,
            password=password,
            security=security,
            max_samples_per_episode=max_samples_per_episode,
            obs_preprocessor=obs_preprocessor,
            standalone=args.test,
        )
        rw.run(test_episode_interval=10)
    elif args.server:
        import time

        serv = Server(port=server_port, password=password, security=security)
        while True:
            time.sleep(1.0)
