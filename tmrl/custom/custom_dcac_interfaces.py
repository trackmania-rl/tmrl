# third-party imports

# local imports
import tmrl.config.config_constants as cfg
from tmrl.drtac import DcacInterface
import logging

class Tm20rtgymDcacInterface(DcacInterface):
    def get_total_delay_tensor_from_augm_obs_tuple_of_tensors(self, augm_obs_tuple_of_tensors):
        """

        Returns:
            tot_del_tensor (batch, 1)
        """
        logging.info(f"DEV: get_total_delay_tensor_from_augm_obs_tuple_of_tensors: augm_obs_tuple_of_tensors:{augm_obs_tuple_of_tensors}")
        exit()
        # return tot_del_tensor

    def replace_act_buf_in_augm_obs_tuple_of_tensors(self, augm_obs_tuple_of_tensors, act_buf_tuple_of_tensors):
        """
        must return a tensor with replaced action buffer

        Args:
            augm_obs_tuple_of_tensors: (batch, obs_shape)
            act_buf_tuple_of_tensors: (batch, act_buf_len, act_shape) (most recent action at idx 0)
        Returns:
            mod_augm_obs_tuple_of_tensors: (batch, obs_shape)
        """
        # logging.info(f"DEV: replace_act_buf_in_augm_obs_tuple_of_tensors: augm_obs_tuple_of_tensors:{augm_obs_tuple_of_tensors}, act_buf_tuple_of_tensors:{act_buf_tuple_of_tensors}")

        rev_act_buf = reversed(act_buf_tuple_of_tensors)
        mod_augm_obs_tuple_of_tensors = (*augm_obs_tuple_of_tensors[:-cfg.ACT_BUF_LEN], *rev_act_buf)

        # logging.info(f"DEV: replace_act_buf_in_augm_obs_tuple_of_tensors: mod_augm_obs_uple_of_tensors:{mod_augm_obs_tuple_of_tensors}")

        return mod_augm_obs_tuple_of_tensors

    def get_act_buf_tuple_of_tensors_from_augm_obs_tuple_of_tensors(self, augm_obs_tuple_of_tensors):
        """

        Args:
            augm_obs_tuple_of_tensors: (batch, obs_shape)
        Returns:
            act_buf_tuple_of_tensors: (batch, act_buf_len, act_shape) (most recent action at idx 0)
        """
        # logging.info(f"DEV: get_act_buf_tuple_of_tensors_from_augm_obs_tuple_of_tensors: augm_obs_tuple_of_tensors:{augm_obs_tuple_of_tensors}")
        act_buf = tuple(reversed(augm_obs_tuple_of_tensors[-cfg.ACT_BUF_LEN:]))
        # logging.info(f"DEV: get_act_buf_tuple_of_tensors_from_augm_obs_tuple_of_tensors: act_buf:{act_buf}")
        return act_buf

    def get_act_buf_size(self):
        """

        Returns:
            act_buf_size: int
        """

        return cfg.ACT_BUF_LEN

    def get_constant_and_max_possible_delay(self):
        """
        Returns:
            is_constant: (bool) whether the delays are constant
            delay: (int) value of the delay if constant, otherwise this output is ignored
        """
        return True, 1
