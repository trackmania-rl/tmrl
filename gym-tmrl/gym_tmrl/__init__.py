import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='gym-tmrl-v0',
    entry_point='gym_tmrl.envs:TMRLEnv',
)
