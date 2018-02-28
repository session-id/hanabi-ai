from __future__ import print_function

import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self):
        self.max_buffer_size = 100
        self.samples = []

    def store(self, step, state, valid_actions_mask, action, reward, done, new_state):
        if len(self.samples) >= self.max_buffer_size:
            self.samples = self.samples[1:]
        self.samples += [tuple(np.array(x) for x in [state, valid_actions_mask, action, reward, done, new_state])]

    def sample(self, batch_size):
        if batch_size > self.max_buffer_size:
            raise RuntimeError("Batch size ({}) cannot be larger than buffer size ({}).".format(batch_size, self.max_buffer_size))

        print("Num samples:", len(self.samples), batch_size)
        num_samples = min(len(self.samples), batch_size)
        indices = random.sample(range(len(self.samples)), num_samples)
        batch_samples = []
        for index in indices:
            batch_samples.append(self.samples[index])
        states, valid_actions_masks, actions, rewards, done_mask, states_next = zip(*batch_samples)
        batch = {} # set states, actions, rewards, states_next, done_mask
        batch['states'] = np.stack(states)
        batch['valid_actions_mask'] = np.stack(valid_actions_masks)
        batch['actions'] = np.stack(actions)
        batch['rewards'] = np.stack(rewards)
        batch['states_next'] = np.stack(states_next)
        batch['done_mask'] = np.stack(done_mask)
        return batch
