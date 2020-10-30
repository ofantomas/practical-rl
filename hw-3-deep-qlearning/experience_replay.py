import numpy as np
import random
import utils
from prioritized_sampler import PrioritizedSampler


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._maxsize = size
        self._index = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize. 
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        data = (obs_t, action, reward, obs_tp1, done)
        maxsize = self._maxsize
        
        if len(self._storage) < maxsize:
            self._storage.append(data)
        else:
            self._storage[self._index] = data
            self._index = (self._index + 1) % self._maxsize
        

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        storage = self._storage
        batch = random.sample(storage, batch_size)
        state, actions, rewards, next_states, is_done = list(map(np.array, (zip(*batch))))
        
        return state, actions, rewards, next_states, is_done
    
class PrioritizedExperienceReplay(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._maxsize = size
        self._index = 0
        self.sampler = PrioritizedSampler(size)

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, p):
        '''
        Make sure, _storage will not exceed _maxsize. 
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        data = (obs_t, action, reward, obs_tp1, done)
        
        if len(self._storage) < self._maxsize:
            self._storage.append(data)
        else:
            self._storage[self._index] = data
            
        self._index = (self._index + 1) % self._maxsize
        self.sampler.update_priorities(self._index, p)
        

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        indices, weights = self.sampler.sample_indices(batch_size)
        batch = [self._storage[idx] for idx in indices]
        state, actions, rewards, next_states, is_done = list(map(np.array, (zip(*batch))))
        
        return state, actions, rewards, next_states, is_done, weights, indices
    
def make_experience_replay(env, agent, init_size=10**4, size=10**4, check_RAM_steps=100):
    state = env.reset()
    exp_replay = ReplayBuffer(size)
    for i in range(int(init_size // check_RAM_steps)):
        if not utils.is_enough_ram(min_available_gb=0.1):
            print("""
                Less than 100 Mb RAM available. 
                Make sure the buffer size in not too huge.
                Also check, maybe other processes consume RAM heavily.
                """
                 )
            break
        utils.play_and_record(state, agent, env, exp_replay, n_steps=check_RAM_steps)
        if len(exp_replay) == size:
            break
    return exp_replay

def make_PER(env, agent, init_size=10**4, size=10**4, check_RAM_steps=100):
    state = env.reset()
    exp_replay = PrioritizedExperienceReplay(size)
    for i in range(int(init_size // check_RAM_steps)):
        if not utils.is_enough_ram(min_available_gb=0.1):
            print("""
                Less than 100 Mb RAM available. 
                Make sure the buffer size in not too huge.
                Also check, maybe other processes consume RAM heavily.
                """
                 )
            break
        utils.play_and_record_PER(state, agent, env, exp_replay, n_steps=check_RAM_steps)
        if len(exp_replay) == size:
            break
    return exp_replay


