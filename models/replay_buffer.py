import random
from collections import deque
import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.samples = deque(maxlen=buffer_size)

    def store(self, step, state, valid_actions_mask, action, reward, done, new_state):
        '''
        Args
        - step: int
        - state: np.array, shape [state_dim]
        - valid_actions_mask: np.array, shape [num_actions]
        - action: int
        - reward: float
        - done: bool
        - new_state: np.array, shape [state_dim]
        '''
        self.samples.append({
            'state': state,
            'valid_actions_mask': valid_actions_mask,
            'action': action,
            'reward': reward,
            'done': done,
            'new_state': new_state
        })

    def sample(self, batch_size):
        '''
        Returns: batch, dict {str: np.array}, where keys are:
        - 'states'
        - 'valid_actions_mask'
        - 'actions'
        - 'rewards'
        - 'states_next'
        - 'done_mask'
        '''
        # sample with replacement
        batch_samples = random.choices(self.samples, batch_size)

        batch = {
            'states': np.stack(sample['state'] for sample in batch_samples),
            'valid_actions_mask': np.stack(sample['valid_actions_mask'] for sample in batch_samples),
            'actions': np.array(sample['action'] for sample in batch_samples),
            'rewards': np.array(sample['reward'] for sample in batch_samples),
            'states_next': np.stack(sample['state_next'] for sample in batch_samples),
            'done_mask': np.array(sample['done'] for sample in batch_samples)
        }
        return batch