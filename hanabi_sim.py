from __future__ import print_function

import copy
import random
from termcolor import colored

COLOR_TO_TERM = {
  'red': 'red',
  'white': 'white',
  'yellow': 'yellow',
  'green': 'green',
  'blue': 'cyan'
}

class Card:
  '''A Hanabi card, comprised of a number and a string color'''

  def __init__(self, number, color):
    self.number = number
    self.color = color

  def __str__(self):
    return colored(self.number, COLOR_TO_TERM[self.color])
    #return str(self.number) + self.color[0]

  def __hash__(self):
    return hash((self.number, self.color))

  def __repr__(self):
    return self.__str__()

class Deck:
  '''A stack of Hanabi cards.'''

  def __init__(self, cards_and_counts):
    self.cards_set = set()
    for card, count in cards_and_counts.items():
      for i in range(count):
        self.cards_set.add((card, i))

  def draw(self):
    drawn_card = random.sample(self.cards_set, 1)[0]
    self.cards_set.remove(drawn_card)
    return drawn_card[0]

  def __len__(self):
    return len(self.cards_set)

class BaseHanabiGame(object):
  def __init__(self, num_players):
    raise NotImplementedError

  def get_state_vector_size(self):
    raise NotImplementedError

  def get_num_actions(self):
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

  def take_action(self, global_state, action):
    '''
    Input:
      global_state: current_state
      action: integer corresponding to index of action taken
        actions are numbered in blocks as follows. x = # of cards in hand, n = # of players,
        h = # of possible hints:
        0 - x-1: play card
        x - 2x-1: discard card
        2x - 2x+h-1: hints for other player 1
        2x+h - 2x+2h-1: hints for other player 2
        ...
        2x+(n-2)h - 2x+(n-1)h-1: hints for player n-1

    Output:
      new_state: the new global state
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

class HanabiState(object):
  '''Global state for Hanabi'''

  def __init__(self, num_players, cards_per_player, colors, max_number, number_counts,
      max_hint_tokens, starting_bomb_tokens):
    self.num_players = num_players
    self.cards_per_player = cards_per_player
    self.max_hint_tokens = max_hint_tokens
    self.starting_bomb_tokens = starting_bomb_tokens
    self.colors = colors
    self.max_number = max_number
    self.number_counts = number_counts

    cards_and_counts = {}
    for color in colors:
      for number, number_count in zip(range(1, max_number+1), number_counts):
        cards_and_counts[Card(number, color)] = number_count
    self.base_deck = Deck(cards_and_counts)

  def initialize_random(self):
    '''
    Initialize a random starting hand for all players.
    '''
    self.deck = copy.deepcopy(self.base_deck)
    self.player_hands = []
    for player_num in range(self.num_players):
      hand = []
      for i in range(self.cards_per_player):
        hand.append(self.deck.draw())
      self.player_hands.append(hand)
    self.hint_history = []
    self.cur_player = 0
    self.played_numbers = {color: 0 for color in self.colors}
    self.hint_tokens = self.max_hint_tokens
    self.bomb_tokens = self.starting_bomb_tokens
    self.turn_no = 0
    self.turns_until_end = self.num_players

class RegularHanabiGameEasyFeatures(object):
  '''
  The regular Hanabi game, 5 colors.
  Currently returning the easy feature set
  '''

  def __init__(self, num_players):
    self.num_players = num_players
    self.cards_per_player = REGULAR_HANABI_CARDS_PER_PLAYER[num_players]
    self.colors = ['red', 'white', 'blue', 'green', 'yellow']
    self.num_colors = len(self.colors)
    self.max_number = 5
    self.number_counts = [3, 2, 2, 2, 1]
    self.max_hint_tokens = 8
    self.starting_bomb_tokens = 3

  def get_start_state(self):
    state = HanabiState(self.num_players, self.cards_per_player, self.colors, self.max_number,
      self.number_counts, self.max_hint_tokens, self.starting_bomb_tokens)
    state.initialize_random()
    return state

  def get_action_names(self, global_state):
    action_names = []
    # Play own cards
    for i in range(self.cards_per_player):
      action_names.append("Play slot {}".format(i))
    # Discard own cards
    for i in range(self.cards_per_player):
      action_names.append("Play slot {}".format(i))
    # Give hints
    for other_player_num in range(1, self.num_players):
      player_id = (global_state.cur_player + other_player_num) % self.num_players
      for number in range(1, self.max_number+1):
        action_names.append("Player {}: hint {}'s".format(player_id, number))
      for color in self.colors:
        action_names.append("Player {}: hint {}"\
          .format(player_id, colored(color, COLOR_TO_TERM[color])))
    return action_names

  def get_valid_actions(self, global_state):
    valid_actions = set()
    action_num = 0
    # Play own cards
    for i in range(self.cards_per_player):
      valid_actions.add(action_num)
      action_num += 1
    # Discard own cards
    for i in range(self.cards_per_player):
      valid_actions.add(action_num)
      action_num += 1
    # Give hints
    if global_state.hint_tokens > 0:
      for other_player_num in range(1, self.num_players):
        player_id = (global_state.cur_player + other_player_num) % self.num_players
        player_hand = global_state.player_hands[player_id]
        for number in range(1, self.max_number+1):
          if any(card.number == number for card in player_hand):
            valid_actions.add(action_num)
          action_num += 1
        for color in self.colors:
          if any(card.color == color for card in player_hand):
            valid_actions.add(action_num)
          action_num += 1
    return valid_actions

  def get_num_actions(self):
    return 2 * self.cards_per_player + (self.num_players - 1) * (self.num_colors + self.max_number)

  def take_action(self, action):
    pass