import torch
from torch import nn
from torch.nn import ModuleList

from tmrl.custom.models.MLPActorCritic import MLPQFunction, SquashedGaussianMLPActor


class REDQMLPActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, n=10
    ):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        # act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(
            observation_space, action_space, hidden_sizes, activation
        )
        self.n = n
        self.qs = ModuleList(
            [
                MLPQFunction(observation_space, action_space, hidden_sizes, activation)
                for _ in range(self.n)
            ]
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.cpu().numpy()
