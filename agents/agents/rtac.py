from copy import deepcopy
from dataclasses import dataclass
from functools import reduce

import torch
from agents.envs import AvenueEnv
from torch.nn.functional import mse_loss

import agents.sac
from agents.memory import Memory
from agents.nn import no_grad, exponential_moving_average, PopArt
from agents.util import partial
from agents.rtac_models import ConvRTAC, ConvDouble


@dataclass(eq=0)
class Agent(agents.sac.Agent):
    Model: type = agents.rtac_models.MlpDouble
    loss_alpha: float = 0.2

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = Memory(self.memory_size, self.batchsize, device)

        self.is_training = False

    def train(self):
        obs, actions, rewards, next_obs, terminals = self.memory.sample()

        new_action_distribution, _, hidden = self.model(obs)
        new_actions = new_action_distribution.rsample()
        new_actions_log_prob = new_action_distribution.log_prob(new_actions)

        # critic loss
        _, next_value_target, _ = self.model_target((next_obs[0], new_actions.detach()))
        next_value_target = reduce(torch.min, next_value_target)
        next_value_target = self.outputnorm_target.unnormalize(next_value_target)

        reward_components = torch.stack((
            self.reward_scale * rewards,
            - self.entropy_scale * new_actions_log_prob.detach(),
        ), dim=1)

        value_target = reward_components + (1. - terminals[:, None]) * self.discount * next_value_target
        # TODO: is it really that helpful/necessary to do the outnorm update here and to recompute the values?
        value_target = self.outputnorm.update(value_target)
        values = tuple(c(h) for c, h in zip(self.model.critic_output_layers, hidden))  # recompute values (weights changed)

        assert values[0].shape == value_target.shape and not value_target.requires_grad
        loss_critic = sum(mse_loss(v, value_target) for v in values)

        # actor loss
        _, next_value, _ = self.model_nograd((next_obs[0], new_actions))
        next_value = reduce(torch.min, next_value)
        new_value = (1. - terminals[:, None]) * self.discount * self.outputnorm.unnormalize(next_value)
        new_value[:, -1] -= self.entropy_scale * new_actions_log_prob
        assert new_value.shape == (self.batchsize, 2)
        loss_actor = - self.outputnorm.normalize_sum(new_value.sum(1)).mean()  # normalize_sum preserves relative scale

        # update model
        self.optimizer.zero_grad()
        loss_total = self.loss_alpha * loss_actor + (1 - self.loss_alpha) * loss_critic
        loss_total.backward()
        self.optimizer.step()

        # update target model and normalizers
        exponential_moving_average(self.model_target.parameters(), self.model.parameters(), self.target_update)
        exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)

        return dict(
            loss_total=loss_total.detach(),
            loss_critic=loss_critic.detach(),
            loss_actor=loss_actor.detach(),
            outputnorm_reward_mean=self.outputnorm.mean[0],
            outputnorm_entropy_mean=self.outputnorm.mean[-1],
            outputnorm_reward_std=self.outputnorm.std[0],
            outputnorm_entropy_std=self.outputnorm.std[-1],
            memory_size=len(self.memory),
            # entropy_scale=self.entropy_scale
        )


# AvenueAgent = partial(
#     Agent,
#     entropy_scale=0.05,
#     lr=0.0002,
#     memory_size=500000,
#     batchsize=100,
#     training_steps=1 / 4,
#     start_training=10000,
#     Model=partial(ConvDouble)
# )
#
# if __name__ == "__main__":
#     from agents import Training, run
#     from agents import rtac_models
#
#     RtacTest = partial(
#         Training,
#         epochs=3,
#         rounds=5,
#         steps=500,
#         Agent=partial(Agent, device='cpu', memory_size=1000000, start_training=256, batchsize=4),
#         Env=partial(id="Pendulum-v0", real_time=True),
#         # Env=partial(id="HalfCheetah-v2", real_time=True),
#     )
#
#     RtacAvenueTest = partial(
#         Training,
#         epochs=3,
#         rounds=5,
#         steps=300,
#         Agent=partial(AvenueAgent, device='cpu', start_training=256, batchsize=4, Model=rtac_models.ConvSeparate),
#         Env=partial(AvenueEnv, real_time=True),
#         Test=partial(number=0),  # laptop can't handle more than that
#     )
#
#     run(RtacTest)
# # run(Rtac_Avenue_Test)
