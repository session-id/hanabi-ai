class Card:
  '''A Hanabi card, comprised of a number and a string color'''

  def __init__(self, number, color):
    self.number = number
    self.color = color

  def __str__(self):
    return self.number + self.color[0]

class HanabiState(Object):
  def __init__(self, ...):
    self.cur_player = 0

class BaseHanabiGame(object):
  def __init__(self, num_players):
    raise NotImplementedError

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

REGULAR_HANABI_CARDS_PER_PLAYER = {
  2: 5,
  3: 5,
  4: 4,
  5: 4
}

class RegularHanabiGame(object):
  '''The regular Hanabi game, 5 colors.'''

  def __init__(self, num_players):
    self.cards_per_player = REGULAR_HANABI_CARDS_PER_PLAYER[num_players]
    self.num_colors = 5
    self.colors = ['red', 'white', 'blue', 'green', 'yellow']
    self.max_hint_tokens = 8
    self.hint_tokens = 8
    self.bomb_tokens = 3