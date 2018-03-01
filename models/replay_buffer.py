import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.max_buffer_size = buffer_size
        self.samples = []

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
        if len(self.samples) >= self.max_buffer_size:
            self.samples = self.samples[1:]

        self.samples.append({
            'state': state,
            'valid_actions_mask': valid_actions_mask,
            'action': action,
            'reward': reward,
            'done': done,
            'state_next': new_state
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
        num_samples = min(batch_size, len(self.samples))
        # sample without replacement
        batch_samples = random.sample(self.samples, k=num_samples)

        batch = {
            'states': np.stack([sample['state'] for sample in batch_samples]),
            'valid_actions_mask': np.stack([sample['valid_actions_mask'] for sample in batch_samples]),
            'actions': np.array([sample['action'] for sample in batch_samples]),
            'rewards': np.array([sample['reward'] for sample in batch_samples]),
            'states_next': np.stack([sample['state_next'] for sample in batch_samples]),
            'done_mask': np.array([sample['done'] for sample in batch_samples])
        }
        return batch
