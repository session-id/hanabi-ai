import random

class ReplayBuffer(object):
    def __init__(self):
        self.buffer_size = 100
        self.samples = []

    def store(self, step, state, action, reward, done, new_state):
        if len(self.samples) >= self.buffer_size:
            self.samples = self.samples[1:]
        self.samples += [tuple(np.array(x) for x in [state, action, reward, done, new_state])]

    def sample(self, batch_size):
        if batch_size > self.buffer_size:
            raise RuntimeError("Batch size ({}) cannot be larger than buffer size ({}).".format(batch_size, self.buffer_size))

        indices = random.sample(range(self.buffer_size), batch_size)
        batch_samples = []
        for index in indices:
            batch_samples.append(self.samples[index])
        states, actions, rewards, states_next, done_mask = zip(*batch_samples)
        batch = {} # set states, actions, rewards, states_next, done_mask
        batch['states'] = states
        batch['actions'] = actions
        batch['rewards'] = rewards
        batch['states_next'] = states_next
        batch['done_mask'] = done_mask
        return batch
