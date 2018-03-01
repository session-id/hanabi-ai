from __future__ import print_function

import hanabi_sim as hs
import random

random.seed(1338)

# deck = hs.Deck({'1r': 3, '5b': 1})
# print(deck.cards_set)
# while len(deck) > 0:
#   print(len(deck), deck.draw())

# print('')

# state = hs.HanabiState(3, 5, ['r', 'b', 'w', 'g', 'y'], 5, [3, 2, 2, 2, 1], 8, 3)
# state.initialize_random()
# print("Deck:")
# print(state.deck.cards_set)
# print("Player hands:")
# print(state.player_hands)

# print('')

def print_action_display(state):
  print("Valid actions:")
  valid_actions = sorted(list(game.get_valid_actions(state)))
  action_names = game.get_action_names(state)
  for action in valid_actions:
    print("{:2}, {}".format(action, action_names[action]))

game = hs.RegularHanabiGameEasyFeatures(3)
state = game.get_start_state()
# assert len(game.get_state_vector(state)) == game.get_state_vector_size()
# for i in range(3):
#   state, reward, done = game.take_action(state, 5)
# print(state.player_hands)
# state, reward, done = game.take_action(state, 10)
# print(state.player_hands)
# state, reward, done = game.take_action(state, 13)
# print(state.player_hands)
# state, reward, done = game.take_action(state, 28)
# print(state.player_hands)
# state, reward, done = game.take_action(state, 24)
# print(state.player_hands)
# state, reward, done = game.take_action(state, 0)
# print(state.player_hands)
# print("Reward: {}, Done: {}".format(reward, done))
# print("Bomb tokens: {}, Hint tokens: {}".format(state.bomb_tokens, state.hint_tokens))
# print(state.get_pretty_board())
# print(sorted(state.deck.cards()))
# print(game.get_state_vector(state))

while True:
  print("Cards remaining in deck: {}, Turns until end: {}, Hint tokens: {}".format(len(state.deck), state.turns_until_end, state.hint_tokens))
  print(game.get_state_vector(state))
  state, reward, done = game.take_action(state, 5)
  if done:
    break
