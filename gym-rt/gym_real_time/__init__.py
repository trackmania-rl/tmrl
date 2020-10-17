from gym_real_time.envs.real_time_env import GymRealTimeInterface
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='gym-rt-v0',
    entry_point='gym_real_time.envs:RealTimeEnv',
)
