class ReplayBuffer(object):
    def __init__(self):
        raise NotImplementedError

    def store(step, state, valid_actions_mask, action, reward, done, new_state):
        raise NotImplementedError

    def sample(self):
        '''
        Returns: batch, dict {str: np.array}, where keys are:
        - 'states'
        - 'actions'
        - 'rewards'
        - 'states_next'
        - 'valid_actions_mask'
        - 'done_mask'
        '''
        raise NotImplementedError