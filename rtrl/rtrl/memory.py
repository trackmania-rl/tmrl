from random import randint
from rtrl.util import collate


class Memory:
  keep_reset_transitions: int = 0

  def __init__(self, memory_size, batchsize, device):
    self.device = device
    self.batchsize = batchsize
    self.capacity = memory_size
    self.memory = []  # list is much faster to index than deque for big sizes

    self.last_observation = None
    self.last_action = None

  def append(self, r, done, info, obs, action):
    if self.last_observation is not None:

      if self.keep_reset_transitions:
        store = True
      else:
        # info["reset"] = True means the episode reset shouldn't be treated as a true terminal state
        store = not info.get('TimeLimit.truncated', False) and not info.get('reset', False)

      if store:
        self.memory.append((self.last_observation, self.last_action, r, obs, done))

    self.last_observation = obs
    self.last_action = action

    # remove old entries if necessary (delete generously so we don't have to do it often)
    if len(self.memory) > self.capacity:
      del self.memory[:self.capacity // 100 + 1]
    return self

  def __len__(self):
    return len(self.memory)

  def __getitem__(self, item):
    return self.memory[item]

  def sample_indices(self):
    return (randint(0, len(self.memory) - 1) for _ in range(self.batchsize))

  def sample(self, indices=None):
    indices = self.sample_indices() if indices is None else indices
    batch = [self.memory[idx] for idx in indices]
    batch = collate(batch, self.device)
    return batch


