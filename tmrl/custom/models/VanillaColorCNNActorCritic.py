from tmrl.custom.models.model_blocks import remove_colors
from tmrl.custom.models.VanillaCNNActorCritic import (
    SquashedGaussianVanillaCNNActor,
    VanillaCNNActorCritic,
    VanillaCNNQFunction,
)


class SquashedGaussianVanillaColorCNNActor(SquashedGaussianVanillaCNNActor):
    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, test=False, with_logprob=True)


class VanillaColorCNNQFunction(VanillaCNNQFunction):
    def forward(self, obs, act):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, act)


class VanillaColorCNNActorCritic(VanillaCNNActorCritic):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # build policy and value functions
        self.actor = SquashedGaussianVanillaColorCNNActor(observation_space, action_space)
        self.q1 = VanillaColorCNNQFunction(observation_space, action_space)
        self.q2 = VanillaColorCNNQFunction(observation_space, action_space)
