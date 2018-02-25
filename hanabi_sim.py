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

class Card(object):
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

class Deck(object):
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

class HintedCard(object):
  def __init__(self, card):
    self.card = card
    self.color_hint = False
    self.number_hint = False

  @property
  def number(self):
    return self.card.number

  @property
  def color(self):
    return self.card.color

  def __str__(self):
    if self.number_hint and self.color_hint:
      extra = '*'
    elif self.number_hint:
      extra = '#'
    elif self.color_hint:
      extra = 'c'
    else:
      extra = ' '
    return str(self.card) + extra

  def __repr__(self):
    return self.__str__()

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
        hand.append(HintedCard(self.deck.draw()))
      self.player_hands.append(hand)
    self.hint_history = []
    self.cur_player = 0
    self.played_numbers = {color: 0 for color in self.colors}
    self.hint_tokens = self.max_hint_tokens
    self.bomb_tokens = self.starting_bomb_tokens
    self.turn_no = 0
    self.turns_until_end = self.num_players

  def add_hint_token(self):
    self.hint_tokens = min(self.max_hint_tokens, self.hint_tokens + 1)

  def advance_player(self):
    self.cur_player = (self.cur_player + 1) % self.num_players

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

  def take_action(self, state, action):
    done = False
    reward = 0

    # If not needed, we can remove this copying for more efficiency
    state = copy.deepcopy(state)
    state.cur_player = state.cur_player
    if action not in self.get_valid_actions(state):
      raise RuntimeError("Invalid action: {}".format(action))

    # Play own cards
    if action < self.cards_per_player:
      played_index = action
      card_played = state.player_hands[state.cur_player][played_index].card
      if state.played_numbers[card_played.color] + 1 == card_played.number:
        if card_played.number == 5:
          state.add_hint_token()
        state.played_numbers[card_played.color] += 1
        reward = 1
      else:
        state.bomb_tokens -= 1
    # Discard
    elif action < self.cards_per_player * 2:
      played_index = action - self.cards_per_player
      state.add_hint_token()

    # For both Discard and Play, need to redraw and shift
    if action < self.cards_per_player * 2:
      if len(state.deck) > 0:
        drawn_card = HintedCard(state.deck.draw())
        previous_hand = state.player_hands[state.cur_player]
        state.player_hands[state.cur_player] = [drawn_card] + previous_hand[:played_index]\
          + previous_hand[played_index+1:]
      else:
        previous_hand = state.player_hands[state.cur_player]
        state.player_hands[state.cur_player] = previous_hand[:played_index]\
          + previous_hand[played_index+1:]

    # Hint
    if action >= self.cards_per_player * 2:
      hint_index = action - self.cards_per_player * 2
      player_hinted = (state.cur_player + hint_index // (self.max_number + self.num_colors) + 1)\
        % self.num_players
      t = hint_index % (self.max_number + self.num_colors)
      # Number hint
      if t < self.max_number:
        hinted_number = t + 1
        for card in state.player_hands[player_hinted]:
          if card.number == hinted_number:
            card.number_hint = True
      # Color hint
      else:
        hinted_color = self.colors[t - self.max_number]
        for card in state.player_hands[player_hinted]:
          if card.color == hinted_color:
            card.color_hint = True

    if state.turns_until_end <= 0:
      done = True

    if len(state.deck) == 0:
      state.turns_until_end -= 1

    state.advance_player()
    if state.bomb_tokens == 1:
      done = True

    return state, reward, done