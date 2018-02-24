class HanabiState(Object):
  def __init__(self, ...):
    self.cur_player = 0

class HanabiGame(object):
  def __init__(self, num_players):
    raise NotImplementedError
    self.cards_per_player = 5 # TODO: Change based on num_players
    self.num_suits = 5
    self.suits = ['red', 'white', 'blue', 'green', 'yellow']
    self.max_hint_tokens = 8
    self.hint_tokens = 8
    self.bomb_tokens = 3

  def get_state_vector(self, global_state):
    '''
    Output:
      state_vector: a vector containing the feature vector corresponding to the game state
        as seen by the provided player
    '''
    raise NotImplementedError

  def get_valid_actions(self, global_state):
    '''
    Output:
      valid_actions: a set containing the indices of valid actions
    '''
    raise NotImplementedError

  def take_action(self, action):
    '''
    Input:
      action: integer corresponding to index of action taken

    Output:
      global state: the new global state
      reward: the reward from performing action
      done: a boolean indicating if the episode has finished
    '''
    raise NotImplementedError
    return global_state, reward, done