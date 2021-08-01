# standard library imports
import itertools
from collections import deque
from random import sample

# third-party imports
import gym
from gym.spaces import Discrete, Tuple
import logging

class RandomDelayWrapper(gym.Wrapper):
    """
    Wrapper for any non-RTRL environment modelling random observation and action delays
    Note that you can access most recent action known to be applied with past_actions[action_delay + observation_delay]
    NB: action_delay represents action channel delay + number of time-steps for which it has been applied
        The brain only needs this information to identify the action that was being executed when the observation was captured
        (the duration for which it had already been executed is irrelevant in the Markov assumption)
    Kwargs:
        obs_delay_range: range in which observation delays are sampled
        act_delay_range: range in which action delays are sampled
        instant_rewards: bool (default True): whether to send instantaneous step rewards (True) or delayed rewards (False)
        initial_action: action (default None): action with which the action buffer is filled at reset() (if None, sampled in the action space)
    """
    def __init__(self, env, obs_delay_range=range(0, 8), act_delay_range=range(0, 2), instant_rewards: bool = False, initial_action=None, skip_initial_actions=False):
        super().__init__(env)
        self.wrapped_env = env
        assert not instant_rewards, 'instant_rewards is depreciated. it was an ill-defined concept'
        self.obs_delay_range = obs_delay_range
        self.act_delay_range = act_delay_range

        self.observation_space = Tuple((
            env.observation_space,  # most recent observation
            Tuple([env.action_space] * (obs_delay_range.stop + act_delay_range.stop - 1)),  # action buffer
            Discrete(obs_delay_range.stop),  # observation delay int64
            Discrete(act_delay_range.stop),  # action delay int64
        ))

        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions
        self.past_actions = deque(maxlen=obs_delay_range.stop + act_delay_range.stop)
        self.past_observations = deque(maxlen=obs_delay_range.stop)
        self.arrival_times_actions = deque(maxlen=act_delay_range.stop)
        self.arrival_times_observations = deque(maxlen=obs_delay_range.stop)

        self.t = 0
        self.done_signal_sent = False
        self.current_action = None
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.

    def reset(self, **kwargs):
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.done_signal_sent = False
        first_observation = super().reset(**kwargs)

        # fill up buffers
        self.t = -(self.obs_delay_range.stop + self.act_delay_range.stop)  # this is <= -2
        while self.t < 0:
            act = self.action_space.sample() if self.initial_action is None else self.initial_action
            self.send_action(act)
            self.send_observation((first_observation, 0., False, {}, 0))
            self.t += 1
        self.receive_action()  # an action has to be applied

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        # logging.debug(" end of reset ---")
        # logging.debug(f" self.past_actions:{self.past_actions}")
        # logging.debug(f" self.past_observations:{self.past_observations}")
        # logging.debug(f" self.arrival_times_actions:{self.arrival_times_actions}")
        # logging.debug(f" self.arrival_times_observations:{self.arrival_times_observations}")
        # logging.debug(f" self.t:{self.t}")
        # logging.debug(" ---")
        return received_observation

    def step(self, action):
        """
        When action delay is 0 and observation delay is 0, this is equivalent to the RTRL setting
        (The inference time is NOT considered part of the action_delay)
        """

        # at the brain
        self.send_action(action)

        # at the remote actor
        if self.t < self.act_delay_range.stop and self.skip_initial_actions:
            # do nothing until the brain's first actions arrive at the remote actor
            self.receive_action()
        elif self.done_signal_sent:
            # just resend the last observation until the brain gets it
            self.send_observation(self.past_observations[0])
        else:
            m, r, d, info = self.env.step(self.current_action)  # before receive_action: rtrl setting with 0 delays
            cur_action_age = self.receive_action()
            self.cum_rew_actor += r
            self.done_signal_sent = d
            self.send_observation((m, self.cum_rew_actor, d, info, cur_action_age))

        # at the brain again
        m, cum_rew_actor_delayed, d, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed

        # logging.debug(" end of step ---")
        # logging.debug(f" self.past_actions:{self.past_actions}")
        # logging.debug(f" self.past_observations:{self.past_observations}")
        # logging.debug(f" self.arrival_times_actions:{self.arrival_times_actions}")
        # logging.debug(f" self.arrival_times_observations:{self.arrival_times_observations}")
        # logging.debug(f" self.t:{self.t}")
        # logging.debug(" ---")
        self.t += 1
        return m, r, d, info

    def send_action(self, action):
        """
        Appends action to the left of self.past_actions
        Simulates the time at which it will reach the agent and stores it on the left of self.arrival_times_actions
        """
        # at the brain
        delay, = sample(self.act_delay_range, 1)
        self.arrival_times_actions.appendleft(self.t + delay)
        self.past_actions.appendleft(action)

    def receive_action(self):
        """
        Looks for the last created action that has arrived before t at the agent
        NB: since this is the most recently created action that the agent got, this is the one currently being applied
        Returns:
            applied_action: int: the index of the action currently being applied
        """
        applied_action = next(i for i, t in enumerate(self.arrival_times_actions) if t <= self.t)
        self.current_action = self.past_actions[applied_action]
        return applied_action

    def send_observation(self, obs):
        """
        Appends obs to the left of self.past_observations
        Simulates the time at which it will reach the brain and appends it in self.arrival_times_observations
        """
        # at the remote actor
        delay, = sample(self.obs_delay_range, 1)
        self.arrival_times_observations.appendleft(self.t + delay)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        """
        Looks for the last created observation at the agent/observer that reached the brain at time t
        NB: since this is the most recently created observation that the brain got, this is the one currently being considered as the last observation
        Returns:
            augmented_obs: tuple:
                m: object: last observation that reached the brain
                past_actions: tuple: the history of actions that the brain sent so far
                observation_delay: int: number of micro time steps it took the last observation to travel from the agent/observer to the brain
                action_delay: int: action travel delay + number of micro time-steps for which the action has been applied at the agent
            r: float: delayed reward corresponding to the transition that created m
            d: bool: delayed done corresponding to the transition that created m
            info: dict: delayed info corresponding to the transition that created m
        """
        # at the brain
        observation_delay = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)
        m, r, d, info, action_delay = self.past_observations[observation_delay]
        return (m, tuple(itertools.islice(self.past_actions, 0, self.past_actions.maxlen - 1)), observation_delay, action_delay), r, d, info


class UnseenRandomDelayWrapper(RandomDelayWrapper):
    """
    Wrapper that translates the RandomDelayWrapper back to the usual RL setting
    Use this wrapper to see what happens to vanilla RL algorithms facing random delays
    """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        t = super().reset(**kwargs)  # t: (m, tuple(self.past_actions), observation_delay, action_delay)
        return t[0]

    def step(self, action):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), observation_delay, action_delay)
        return (t[0], *aux)
