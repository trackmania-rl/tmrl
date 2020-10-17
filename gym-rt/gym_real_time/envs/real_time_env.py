from gym import Env
import gym.spaces as spaces
import time
from collections import deque
from threading import Thread, Lock


# General Interface class ==============================================================================================
# All user-defined interfaces should be subclasses of GymRealTimeInterface

# TODO: implement a dummy example with the random delay wrapper on e.g. Pendulum

class GymRealTimeInterface:
    """
    Implement this class for your application
    """
    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing
        Args:
            control: np.array of the dimension of the action-space
        """
        # if control is not None:
        #     ...

        raise NotImplementedError

    def reset(self):
        """
        Returns:
            obs: must be a list of numpy arrays
        """
        # return obs

        raise NotImplementedError

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())

    def get_obs_rew_done(self):
        """
        Returns:
            obs: list of numpy arrays
            rew: scalar
            done: boolean
        """
        # return obs, rew, done

        raise NotImplementedError

    def get_observation_space(self):
        """
        Returns:
            observation_space: gym.spaces.Tuple
        """
        # return spaces.Tuple(...)

        raise NotImplementedError

    def get_action_space(self):
        """
        Returns:
            action_space: gym.spaces.Box
        must be a Box
        """
        # return spaces.Box(...)

        raise NotImplementedError

    def get_default_action(self):
        """
        Returns:
            default_action: numpy array of the dimension of the action space
        initial action at episode start
        """
        # return np.array([...], dtype='float32')

        raise NotImplementedError


# General purpose environment: =========================================================================================

DEFAULT_CONFIG_DICT = {
    "interface": GymRealTimeInterface,  # TODO: change this for a dummy interface
    "time_step_duration": 0.05,
    "start_obs_capture": 0.04,
    "time_step_timeout_factor": 1.0,
    "ep_max_length": 1000,  # np.inf
    "real_time": True,
    "async_threading": True,
    "act_in_obs": True,
    "act_buf_len": 1,
    "reset_act_buf": True,
    "benchmark": True
}


class RealTimeEnv(Env):
    def __init__(self, config=DEFAULT_CONFIG_DICT):
        """
        :param interface: (callable) external interface class (required)
        :param ep_max_length: (int) the max length of each episodes in timesteps
        :param real_time: bool: whether to use the RTRL setting
        :param async_threading: bool (optional, default: True): whether actions are executed asynchronously in the RTRL setting.
            Typically this is useful for the real world and for external simulators
        :param time_step_duration: float (optional, default 0.0): seconds slept after apply_action() (~ time-step duration)
        :param act_in_obs: bool (optional, default True): whether to augment the observation with the action buffer (DCRL)
        :param act_buf_len: int (optional, default 1): length of the action buffer (DCRL)
        :param default_action: float (optional, default None): default action to append at reset when the previous is True
        :param act_prepro_func: function (optional, default None): function that maps the action input to the actual applied action
        :param obs_prepro_func: function (optional, default None): function that maps the observation output to the actual returned observation
        :param reset_act_buf: bool (optional, defaut True): whether action buffer should be re-initialized at reset
        """
        # interface:
        interface_cls = config["interface"]
        self.interface = interface_cls()

        # config variables:
        self.act_prepro_func: callable = config["act_prepro_func"] if "act_prepro_func" in config else None
        self.obs_prepro_func = config["obs_prepro_func"] if "obs_prepro_func" in config else None
        self.ep_max_length = config["ep_max_length"]

        self.time_step_duration = config["time_step_duration"] if "time_step_duration" in config else 0.0
        self.time_step_timeout_factor = config["time_step_timeout_factor"] if "time_step_timeout_factor" in config else 1.0
        self.start_obs_capture = config["start_obs_capture"] if "start_obs_capture" in config else 1.0
        self.time_step_timeout = self.time_step_duration * self.time_step_timeout_factor  # time after which elastic time-stepping is dropped
        self.real_time = config["real_time"]
        self.async_threading = config["async_threading"] if "async_threading" in config else True
        self.__t_start = time.time()  # beginning of the time-step
        self.__t_co = time.time()  # time at which observation starts being captured during the time step
        self.__t_end = time.time()  # end of the time-step
        if not self.real_time:
            self.async_threading = False
        if self.async_threading:
            self._at_thread = Thread(target=None, args=(), kwargs={}, daemon=True)
            self._at_thread.start()  # dummy start for later call to join()

        # observation capture:
        self.__o_lock = Lock()  # lock to retrieve observations asynchronously, acquire to access the following:
        self.__obs = None
        self.__rew = None
        self.__done = None
        self.__o_set_flag = False

        # environment benchmark:
        self.benchmark = config["benchmark"] if "benchmark" in config else False
        self.__b_lock = Lock()
        self.__b_obs_capture_duration = 0.0
        self.running_average_factor = config["running_average_factor"] if "running_average_factor" in config else 0.1

        self.act_in_obs = config["act_in_obs"] if "act_in_obs" in config else True
        self.act_buf_len = config["act_buf_len"] if "act_buf_len" in config else 1
        self.act_buf = deque(maxlen=self.act_buf_len)
        self.reset_act_buf = config["reset_act_buf"] if "reset_act_buf" in config else True
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.current_step = 0
        self.initialized = False
        # state variables:
        self.default_action = self.interface.get_default_action()
        self.last_action = self.default_action

    def _update_timestamps(self):
        """
        This is called at the beginning of each time-step
        If the previous time-step has timed out, the beginning of the time-step is set to now
        Otherwise, the beginning of the time-step is the beginning of the previous time-step + the time-step duration
        The observation starts being captured start_obs_capture_factor time-step after the beginning of the time-step
            observation capture can exceed the time-step, it is fine, but be cautious with timeouts
        It is recommended to draw a time diagram of your system
            action computation and observation capture can be performed in parallel
        """
        now = time.time()
        if now < self.__t_end + self.time_step_timeout:  # if either still in the previous time-step of within its allowed elasticity
            self.__t_start = self.__t_end  # the new time-step starts when the previous time-step is supposed to finish or to have finished
        else:  # if after the allowed elasticity
            print(f"INFO: time-step timed out. Elapsed since last time-step: {now - self.__t_end}")
            self.__t_start = now  # the elasticity is broken and reset (this should happen only after 'pausing' the environment)
        self.__t_co = self.__t_start + self.start_obs_capture  # update time at which observation should be retrieved
        self.__t_end = self.__t_start + self.time_step_duration  # update time at which the new time-step should finish

    def _join_thread(self):
        """
        This is called at the beginning of every user-side API functions (step(), reset()...) for thread safety
        This ensures that the previous time-step is completed when starting a new one
        """
        if self.async_threading:
            self._at_thread.join()

    def _run_time_step(self, *args, **kwargs):
        """
        This is what must be called in step() to apply an action
        Call this with the args and kwargs expected by self.__send_act_get_obs_and_wait()
        This in turn calls self.__send_act_get_obs_and_wait()
        In action-threading, self.__send_act_get_obs_and_wait() is called in a new Thread
        """
        if not self.async_threading:
            self.__send_act_get_obs_and_wait(*args, **kwargs)
        else:
            self._at_thread = Thread(target=self.__send_act_get_obs_and_wait, args=args, kwargs=kwargs, daemon=True)
            self._at_thread.start()

    def _initialize(self):
        """
        This is called at first reset() for rllib compatibility
        All costly initializations should be performed here
        This allows creating a dummy environment for retrieving action space and observation space without performing these initializations
        """
        self.init_action_buffer()
        self.initialized = True

    def _get_action_space(self):
        return self.interface.get_action_space()

    def _get_observation_space(self):
        t = self.interface.get_observation_space()
        if self.act_in_obs:
            t = spaces.Tuple((*t.spaces, *((self._get_action_space(),) * self.act_buf_len)))
        return t

    def __send_act_get_obs_and_wait(self, action):
        """
        This function applies the control and launches observation capture at the right timestamp
        !: only one such function must run in parallel (always join thread)
        """
        act = self.act_prepro_func(action) if self.act_prepro_func else action
        self.interface.send_control(act)
        self._update_timestamps()
        now = time.time()
        if now < self.__t_co:  # wait until it is time to capture observation
            time.sleep(self.__t_co - now)
        self.__update_obs_rew_done()  # capture observation
        now = time.time()
        if now < self.__t_end:  # wait until the end of the time-step
            time.sleep(self.__t_end - now)

    def __update_obs_rew_done(self):
        """
        Captures o, r, d asynchronously
        Returns:
            observation of this step()
        """
        if self.benchmark:
            t1 = time.time()
        self.__o_lock.acquire()
        o, r, d = self.interface.get_obs_rew_done()
        if not d:
            d = (self.current_step >= self.ep_max_length)
        elt = o
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        elt = tuple(elt)
        self.__obs, self.__rew, self.__done = elt, r, d
        self.__o_set_flag = True
        self.__o_lock.release()
        if self.benchmark:
            t = time.time() - t1
            self.__b_lock.acquire()
            self.__b_obs_capture_duration = (1.0 - self.running_average_factor) * self.__b_obs_capture_duration + self.running_average_factor * t
            self.__b_lock.release()

    def _retrieve_obs_rew_done(self):
        """
        Waits for new available o r d and retrieves them
        """
        c = True
        while c:
            self.__o_lock.acquire()
            if self.__o_set_flag:
                elt, r, d = self.__obs, self.__rew, self.__done
                self.__o_set_flag = False
                c = False
            self.__o_lock.release()
        if self.act_in_obs:
            elt = tuple((*elt, *tuple(self.act_buf),))
        return elt, r, d

    def init_action_buffer(self):
        for _ in range(self.act_buf_len):
            self.act_buf.append(self.default_action)

    def reset(self):
        """
        Use reset() to reset the environment
        Returns:
            obs
        """
        self._join_thread()
        if not self.initialized:
            self._initialize()
        self.current_step = 0
        elt = self.interface.reset()
        if self.reset_act_buf:
            self.init_action_buffer()
        if self.act_in_obs:
            elt = elt + list(self.act_buf)
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        elt = tuple(elt)
        if self.real_time:
            self._run_time_step(self.default_action)
        return elt

    def step(self, action):
        """
        Call this function to perform a step
        Args:
            action: numpy.array: control value
        Returns:
            obs, rew, done, info

        CAUTION: the drone is only 'paused' at the end of the episode (the entire episode must be rolled out before optimizing if the optimization is synchronous)
        """
        self._join_thread()
        self.current_step += 1
        self.act_buf.append(action)
        if not self.real_time:
            self._run_time_step(action)
        obs, rew, done = self._retrieve_obs_rew_done()
        info = {}
        if self.real_time:
            self._run_time_step(action)
        if done:
            self.interface.wait()
        return obs, rew, done, info

    def stop(self):
        self._join_thread()

    def wait(self):
        self._join_thread()
        self.interface.wait()

    def benchmarks(self):
        """
        Returns the following running averages when the benchmark option is set:
            - duration of __update_obs_rew_done()
        """
        assert self.benchmark, "The benchmark option is not set. Set benchmark=True in the configuration dictionary of the environment"
        self.__b_lock.acquire()
        res_obs_capture_duration = self.__b_obs_capture_duration
        self.__b_lock.release()
        return res_obs_capture_duration

    def render(self, mode='human'):
        """
        Visually renders the current state of the environment
        """
        self._join_thread()
        print("render")
