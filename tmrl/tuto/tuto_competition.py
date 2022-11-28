"""
=======================================================================
COMPETITION TUTORIAL #1: Custom model and RL algorithm
=======================================================================

In this tutorial, we customize the default TrackMania pipeline.

To submit an entry to the TMRL competition, we essentially need a
trained policy. In TMRL, this is encapsulated in an ActorModule.

Note: this tutorial describes implementing a TrainingAgent in TMRL.
The TMRL framework is relevant if you want to implement RL approaches.
If you plan to try non-RL approaches instead, this is also accepted:
just use the competition Gym environment and do whatever you need,
then, wrap your trained policy in an ActorModule, and submit :)
"""

# Okay folks, we will start by importing useful stuff.

# The constants that are defined in config.json:
import tmrl.config.config_constants as cfg
# High-level constants that are fixed for the competition:
import tmrl.config.config_objects as cfg_obj
# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import partial
# The main TMRL classes:
from tmrl.networking import Server, RolloutWorker, Trainer

# The training class that we will customize with our own training algorithm:
from tmrl.training_offline import TrainingOffline

# And useful external libraries:
import numpy as np


# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# You can change these parameters here directly,
# or you can change them in the config.json file.

# maximum number of training 'epochs':
# training is checkpointed at the end of each 'epoch'
# this is also when training metrics can be logged to wandb
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# number of rounds per 'epoch':
# (training metrics are displayed in the terminal at the end of each round)
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# number of training steps per round:
# (a training step is a call to the train() function that we will define later in this tutorial)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# minimum number of environment steps collected before training starts
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# maximum training steps / env steps ratio:
# (if training becomes faster than this ratio, it will be paused waiting for new samples from the environment)
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# number of training steps between when the Trainer broadcasts policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# number of training steps between when the Trainer updates its replay buffer with the buffer of received samples:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# training device (e.g., "cuda:0"):
# if None, the training device will be selected automatically
device = None

# maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================
# You may want to change the following in advanced applications;
# however, most competitors will not need to change this.
# If interested, read the full TMRL tutorial on GitHub.

# base class of the replay memory:
memory_base_cls = cfg_obj.MEM

# sample preprocessor for data augmentation:
sample_preprocessor = None

# path from where an offline dataset can be loaded:
dataset_path = cfg.DATASET_PATH


# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================
# Competitors CANNOT change the following parameters.
# (Note: For models such as RNNs, you don't need to use imgs_buf_len
# and act_buf_len, but your ActorModule implementation needs to work
# with the observations corresponding to their default values. The rule
# about these history lengths is only here for simplicity. You are
# allowed to hack this within your ActorModule implementation by, e.g.,
# storing your own histories if you like.)

# rtgym environment class (full TrackMania Gym environment):
env_cls = cfg_obj.ENV_CLS

# number of consecutive screenshots (this is part of observations):
imgs_buf_len = cfg.IMG_HIST_LEN

# number of actions in the action buffer (this is part of observations):
act_buf_len = cfg.ACT_BUF_LEN


# =====================================================================
# MEMORY CLASS
# =====================================================================
# Nothing to do here.
# This is the memory class passed to the Trainer.
# If you need a custom memory, change the relevant advanced parameters.
# Custom memories are described in the full TMRL tutorial.

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False,
                     use_dataloader=False,
                     pin_memory=False)


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
# The following constants are from the Spinnup implementation of SAC
# that we simply adapt in this tutorial.
LOG_STD_MAX = 2
LOG_STD_MIN = -20


# Let us import this thing that we are supposed to implement.
from tmrl.actor import ActorModule

# Pytorch and math will be useful too:
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor


# In the full version of the TrackMania 2020 environment, the
# observation-space comprises a history of screenshots. Thus, we need
# Computer Vision layers such as a CNN in our model to process these.
# The observation space also comprises single floats representing speed,
# rpm and gear. We will merge these with the information contained in
# screenshots thanks to an MLP following our CNN layers.


# Here is the MLP:
def mlp(sizes, activation, output_activation=nn.Identity):
    """
    A simple MLP.

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


# This utility computes the dimensionality of CNN feature maps when flattened together:
def num_flat_features(x):
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# This utility computes the dimensionality of the output in a 2D CNN layer:
def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


# Let us now define a module that will be the main building block of both our actor and critic:
class VanillaCNN(nn.Module):
    def __init__(self, q_net):
        """
        Simple CNN model for SAC.

        Args:
            q_net (bool): indicates whether the object is a critic network
        """
        super(VanillaCNN, self).__init__()

        self.q_net = q_net

        # Convolutional layers processing screenshots:
        self.h_out, self.w_out = 64, 64  # We will feed grayscale images of 64 x 64 pixels to our model
        self.conv1 = nn.Conv2d(4, 64, 8, stride=2)
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
        # - the 2 previous actions (2 x 3 floats), important because of the real-time nature of our controller
        # - when the module is the critic, the selected action (3 floats)
        self.mlp_input_features = self.flat_features + 12 if self.q_net else self.flat_features + 9

        # MLP layers:
        # (when using the model as a policy, we will sample from a multivariate gaussian defined later in the tutorial;
        # thus, the output dimensionality is  1 for the critic, and we will define the output layer of policies later)
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        """
        In Pytorch, the forward function is where our neural network computes its output from its input.

        Args:
            x (torch.Tensor): input tensor (i.e., the observation fed to our deep neural network)

        Returns:
            the output of our neural network in the form of a torch.Tensor
        """
        if self.q_net:
            # The critic takes the current action act as additional input
            # act1 and act2 are the actions in the action buffer (see real-time RL):
            speed, gear, rpm, images, act1, act2, act = x
        else:
            # For the policy, we still need the action buffer in observations:
            speed, gear, rpm, images, act1, act2 = x

        # Forward pass of our images in the CNN:
        # (note that the competition environment outputs histories of 4 images,
        # we will stack these images along the channel dimension of our input tensor)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Now we will flatten our output feature map.
        # Let us double-check that our dimensions are what we expect them to be:
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"
        # All good, let us flatten our output feature map:
        x = x.view(-1, flat_features)

        # Finally, we can feed the result along with our float values to the MLP:
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network :)
        return x


# We can now implement this TMRL ActorModule interface that we are supposed to submit for the competition.
# Once trained, TMRL will save it in the TmrlData/weights folder.
class SquashedGaussianVanillaCNNActor(ActorModule):
    """
    Our policy wrapped in the TMRL ActorModule class.

    The only required method is ActorModule.act().
    We also implement a forward() method for our training algorithm.
    (Note: ActorModule is a subclass of torch.Module)  # FIXME change this useless thing
    """
    def __init__(self, observation_space, action_space):
        """
        When implementing __init__, we need to take the observation_space, action_space arguments.
        They are here for convenience in case we want them.

        Args:
            observation_space: observation space of the Gym environment
            action_space: action space of the Gym environment
        """
        # We must also call the superclass __init__:
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]  # dimensionality of actions
        act_limit = action_space.high[0]  # maximum amplitude of actions
        # Our hybrid CNN+MLP policy:
        self.net = VanillaCNN(q_net=False)
        # The policy output layer, which samples actions stochastically in a gaussian, with means...:
        self.mu_layer = nn.Linear(256, dim_act)
        # ... and log standard deviations:
        self.log_std_layer = nn.Linear(256, dim_act)
        # We will squash this within the action space thanks to a tanh final activation:
        self.act_limit = act_limit

    def forward(self, obs, test=False):
        """
        Computes the output action of our policy from the input observation.

        The whole point of deep RL will be to train our policy network such that is outputs relevant actions.
        Training per-se will also rely on a critic network, but this is not part of the trained policy.

        Args:
            obs: the observation from the Gym environment
            test: this is True for test episodes (deployment) and False for training episodes

        Returns:
            the action sampled from our policy
            the log probability of this action (this will be used for SAC)
        """
        # MLP:
        net_out = self.net(obs)
        # means of the multivariate gaussian (action vector)
        mu = self.mu_layer(net_out)
        # standard deviations:
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # action sampling
        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        # log probabilities:
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        # squashing within the action space:
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    def act(self, obs, test=False):
        """
        Computes an action from an observation.

        Args:
            obs (object): the observation
            test (bool): True at test time, False otherwise

        Returns:
            act (numpy.array): the computed action
        """
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.numpy()


# The critic module is straightforward:
class VanillaCNNQFunction(nn.Module):
    """
    Critic module.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)

    def forward(self, obs, act):
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)


# Finally, let us merge this together into an actor-critic module for training.
# Classically, we use two parallel critics to alleviate the overestimation bias.
class VanillaCNNActorCritic(nn.Module):
    """
    Actor-critic module for the SAC algorithm.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()

        # build policy and value functions
        self.actor = SquashedGaussianVanillaCNNActor(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.numpy()


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
# - train(batch): optimizes the model from a batch of RL samples
# - get_actor(): outputs a copy of the current ActorModule
# In this tutorial, we will implement the Soft Actor-Critic algorithm
# by adapting the OpenAI Spinnup implementation to the TMRL library.
class SACTrainingAgent(TrainingAgent):
    """
    Our custom training algorithm (SAC).

    Your implementation must at least pass these three arguments to the superclass.

    Args:
        observation_space (Gym.spaces.Space): observation space (here for your convenience)
        action_space (Gym.spaces.Space): action space (here for your convenience)
        device (str): torch device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)
    """

    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,  # Gym observation space (required argument here for your convenience)
                 action_space=None,  # Gym action space (required argument here for your convenience)
                 device=None,  # Device our TrainingAgent should use for training (required argument)
                 model_cls=MyActorCriticModule,  # an actor-critic module, encapsulating our ActorModule
                 gamma=0.99,  # discount factor
                 polyak=0.995,  # exponential averaging factor for the target critic
                 alpha=0.2,  # fixed (SAC v1) or initial (SAC v2) value of the entropy coefficient
                 lr_actor=1e-3,  # learning rate for the actor
                 lr_critic=1e-3,  # learning rate for the critic
                 lr_entropy=1e-3,  # entropy autotuning coefficient (SAC v2)
                 learn_entropy_coef=True,  # if True, SAC v2 is used, else, SAC v1 is used
                 target_entropy=None):  # if None, the target entropy for SAC v2 is set automatically
        # required arguments passed to the superclass:
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        # custom stuff:
        model = model_cls(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_entropy = lr_entropy
        self.learn_entropy_coef=learn_entropy_coef
        self.target_entropy = target_entropy
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)
        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        """
        Returns the current ActorModule.

        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor

    def train(self, batch):
        """
        Executes a training iteration from batched tensors.

        Args:
            batch: (previous observation, action, reward, new observation, terminated signal, truncated signal)

        Returns:
            logs: Dictionary: a python dictionary of training metrics you wish to log on wandb
        """
        o, a, r, o2, d, _ = batch
        pi, logp_pi = self.model.actor(o)
        loss_alpha = None
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        for p in self.q_params:
            p.requires_grad = False
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha_t * logp_pi - q_pi).mean()
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        for p in self.q_params:
            p.requires_grad = True
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        ret_dict = dict(
            loss_actor=loss_pi.detach(),
            loss_critic=loss_q.detach(),
        )
        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
            ret_dict["entropy_coef"] = alpha_t.item()
        return ret_dict


training_agent_cls = partial(SACTrainingAgent,
                             model_cls=MyActorCriticModule,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.2,
                             lr_actor=1e-3,
                             lr_critic=1e-3,
                             lr_entropy=1e-3,
                             learn_entropy_coef=True,
                             target_entropy=None)


# Trainer instance:

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
    device=device)















training_agent_cls = None

training_cls = partial(TrainingOffline,
                       training_agent_cls=training_agent_cls,
                       epochs=epochs,
                       rounds=rounds,
                       steps=steps,
                       update_buffer_interval=update_buffer_interval,
                       update_model_interval=update_model_interval,
                       max_training_steps_per_env_step=max_training_steps_per_env_step,
                       start_training=start_training,
                       device=device,
                       env_cls=env_cls,
                       memory_cls=memory_cls)

if __name__ == "__main__":
    my_trainer = Trainer(training_cls=training_cls)
