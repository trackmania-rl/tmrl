from typing import Sequence
from math import floor

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from tmrl.core.util import prod

from tmrl.core.jax.util import get_rngs
from tmrl.core.jax.actor import NNXActorModule

import tmrl.config.config_constants as cfg


LOG_STD_MAX = 2
LOG_STD_MIN = -20


# TODO: set Rngs as class attributes for persistence (NNX side effects should be OK)


def mlp(
    sizes: Sequence[int],
    activation = nnx.relu,
    output_activation = nnx.identity,
    dropout: float | Sequence[float] = 0.0,
    layer_norm: bool | Sequence[bool] = False,
    rngs: nnx.Rngs | None = None,
) -> nnx.Sequential:

    rngs = rngs or get_rngs()
    layers = []

    if not isinstance(dropout, (list, tuple)):
        dropout = [dropout, ] * (len(sizes) - 1)
    if not isinstance(layer_norm, (list, tuple)):
        layer_norm = [layer_norm, ] * (len(sizes) - 1)
    if len(dropout) != (len(sizes) - 1) or len(layer_norm) != (len(sizes) - 1):
        raise RuntimeError(f"Invalid argument shapes. sizes:{len(sizes)}, dropout:{len(dropout)}, layer_norm:{len(layer_norm)}")
    for j in range(len(sizes) - 1):
        layers.append(nnx.Linear(sizes[j], sizes[j + 1], rngs=rngs))
        if dropout[j]:
            layers.append(nnx.Dropout(rate=dropout[j], rngs=rngs))
        if layer_norm[j]:
            layers.append(nnx.LayerNorm(sizes[j + 1], rngs=rngs))
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(act)

    return nnx.Sequential(*layers)


class NNXSquashedGaussianMLPActor(NNXActorModule):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nnx.relu,
                 layer_norm=False,
                 rngs: nnx.Rngs = None):
        super().__init__(observation_space, action_space)
        rngs = rngs or get_rngs()
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False
        dim_act = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.net = mlp(sizes=[dim_obs] + list(hidden_sizes),
                       activation=activation,
                       output_activation=activation,
                       layer_norm=layer_norm,
                       rngs=rngs)
        self.mu_layer = nnx.Linear(hidden_sizes[-1], dim_act, rngs=rngs)
        self.log_std_layer = nnx.Linear(hidden_sizes[-1], dim_act, rngs=rngs)

    def __call__(self, obs, test=False, with_logprob=True, epsilon=1e-8, rngs: nnx.Rngs=None):
        """
        Note: this function assumes a batch dimension in obs.
        Obs can be either a simple batched tensor, or a collated tuple of batched tensors.
        """
        print("call")
        print(obs.shape)
        x = jnp.concatenate(obs, axis=-1) if self.tuple_obs else obs.reshape(obs.shape[0], -1)
        print(x.shape)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        # Pre-squash distribution and sample
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            rngs = rngs or get_rngs()
            eps = jax.random.normal(rngs.noise(), mu.shape)
            pi_action = mu + std * eps

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: explanation at https://github.com/openai/spinningup/issues/279
            # explicit log prob calculation
            logp_pi = (-0.5 * ((pi_action - mu) / (std + epsilon))**2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)).sum(axis=-1)
            logp_pi -= (2 * (jnp.log(2) - pi_action - jax.nn.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = jnp.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def act(self, obs, test=False):
        a, _ = self.__call__(obs, test, False)
        res = np.array(a.squeeze())
        if not len(res.shape):
            res = np.expand_dims(res, 0)
        return res


class NNXMLPQFunction(nnx.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nnx.relu,
                 dropout=0.0,
                 layer_norm=False,
                 rngs: nnx.Rngs = None):
        rngs = rngs or get_rngs()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(observation_space.shape)
            self.tuple_obs = False
        act_dim = action_space.shape[0]
        dropout_list = [dropout] * len(hidden_sizes) + [0.0]
        layer_norm_list = [layer_norm] * len(hidden_sizes) + [False]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, dropout=dropout_list, layer_norm=layer_norm_list, rngs=rngs)

    def __call__(self, obs, act):
        x = jnp.concatenate((*obs, act), -1) if self.tuple_obs else jnp.concatenate((obs.reshape(obs.shape[0], -1), act), axis=-1)
        q = self.q(x)
        return q.squeeze(-1)


class NNXREDQMLPActorCritic(nnx.Module):
    """
    By default, this holds 2 critics for SAC.
    Set n to a higher value for REDQ-SAC.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 n=2,
                 hidden_sizes=(256, 256),
                 activation=nnx.relu,
                 critic_dropout=0.0,
                 critic_layer_norm=False,
                 actor_layer_norm=False,
                 rngs: nnx.Rngs = None):
        rngs = rngs or get_rngs()
        self.n = n

        # build policy and value functions
        self.actor = NNXSquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation, layer_norm=actor_layer_norm, rngs=rngs)
        self.qs = nnx.List([
            NNXMLPQFunction(observation_space=observation_space, action_space=action_space, hidden_sizes=hidden_sizes, activation=activation, dropout=critic_dropout, layer_norm=critic_layer_norm, rngs=rngs)
            for _ in range(self.n)
        ])

if __name__ == "__main__":
    import jax.numpy as jnp
    import gymnasium as gym

    env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
    act_space = env.action_space
    obs_space = env.observation_space

    ac = NNXREDQMLPActorCritic(observation_space=obs_space, action_space=act_space)

    # Forward pass
    x = obs_space.sample()

    model = ac.actor

    model.train()
    y = model.act_(x)
    print(y)
    model.eval()
    y = model.act_(x)
    print(y)
    y = model.act_(x)
    print(y)



# # --- CNN helpers ---

# def conv2d_out_dims(conv, h_in, w_in):
#     def out(i, padding, dilation, kernel, stride):
#         return floor((i + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
#     h_out = out(h_in, conv.strides[0], 1, conv.kernel_size[0], conv.strides[0])  # NNX Conv uses strides/kernel_size tuples
#     w_out = out(w_in, conv.strides[1], 1, conv.kernel_size[1], conv.strides[1])
#     return h_out, w_out


# # --- Vanilla CNN ---

# class VanillaCNN(nnx.Module):
#     def __init__(self, q_net, img_height, img_width, img_hist_len, dropout=0.0, layer_norm=False, rngs: nnx.Rngs = None):
#         self.q_net = q_net
#         h, w = img_height, img_width

#         # NNX Conv2d: (in_features, out_features, kernel_size) — channels-last by default
#         self.conv1 = nnx.Conv(img_hist_len, 64, kernel_size=(8, 8), strides=(2, 2), rngs=rngs)
#         h, w = floor((h - 8) / 2 + 1), floor((w - 8) / 2 + 1)
#         self.conv2 = nnx.Conv(64, 64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
#         h, w = floor((h - 4) / 2 + 1), floor((w - 4) / 2 + 1)
#         self.conv3 = nnx.Conv(64, 128, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
#         h, w = floor((h - 4) / 2 + 1), floor((w - 4) / 2 + 1)
#         self.conv4 = nnx.Conv(128, 128, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
#         h, w = floor((h - 4) / 2 + 1), floor((w - 4) / 2 + 1)

#         flat_features = 128 * h * w
#         mlp_input = flat_features + 12 if q_net else flat_features + 9

#         if q_net:
#             mlp_sizes = [mlp_input, 256, 256, 1]
#             drop = [dropout, dropout, 0.0]
#             ln = [layer_norm, layer_norm, False]
#         else:
#             mlp_sizes = [mlp_input, 256, 256]
#             drop = [dropout, dropout]
#             ln = [layer_norm, layer_norm]

#         self.mlp = mlp(mlp_sizes, nnx.relu, dropout=drop, layer_norm=ln, rngs=rngs)

#     def __call__(self, x):
#         if self.q_net:
#             speed, gear, rpm, images, act1, act2, act = x
#         else:
#             speed, gear, rpm, images, act1, act2 = x

#         # NNX Conv expects (batch, H, W, C); permute if images are (batch, C, H, W)
#         images = jnp.transpose(images, (0, 2, 3, 1))

#         x = nnx.relu(self.conv1(images))
#         x = nnx.relu(self.conv2(x))
#         x = nnx.relu(self.conv3(x))
#         x = nnx.relu(self.conv4(x))
#         x = x.reshape(x.shape[0], -1)

#         if self.q_net:
#             x = jnp.concatenate([speed, gear, rpm, x, act1, act2, act], axis=-1)
#         else:
#             x = jnp.concatenate([speed, gear, rpm, x, act1, act2], axis=-1)

#         return self.mlp(x)
