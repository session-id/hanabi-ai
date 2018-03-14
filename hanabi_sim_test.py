from __future__ import print_function

import numpy as np
import random

import hanabi_sim as hs
from hanabi_expert import HanabiExpert

random.seed(1338)
np.random.seed(1338)

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

game = hs.RegularHanabiGameEasyFeatures(2, ['red', 'white'], 3, 3, [3, 2, 2])

def print_action_display(state):
    print("Valid actions:")
    valid_actions = sorted(list(game.get_valid_actions(state)))
    action_names = game.get_action_names(state)
    for action in valid_actions:
        print("{:2}, {}".format(action, action_names[action]))

state = game.get_start_state()
expert = HanabiExpert(game)
state.player_hands[1][0].number_hint = True
state.player_hands[1][1].number_hint = True
state.player_hands[1][2].number_hint = True
state.print_self()
action = expert.get_action(state)
print(action, game.get_action_names(state)[action])

# rewards = []
# num_games = 1000
# for game_num in range(num_games):
#     if game_num % 100 == 0:
#         print(game_num)
#     state = game.get_start_state()
#     game_reward = 0
#     while True:
#         features = game.get_state_vector(state, cheat=True)
#         valid_actions = sorted(list(game.get_valid_actions(state)))
#         if game.get_state_vector(state)[17] == 1:
#             action = 0
#         elif len(valid_actions) > 2:
#             action = valid_actions[2]
#         else:
#             action = 1
#         action = np.random.randint(0, 3)
#         state, reward, done = game.take_action(state, action)
#         game_reward += reward
#         if done:
#             break
#     rewards.append(float(game_reward))
# 
# rewards = np.array(rewards)
# print("Mean:", np.mean(rewards))
# print("Std:", np.std(rewards) / np.sqrt(num_games))

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

# while True:
#     state.print_self()
#     print(np.reshape(game.get_state_vector(state, cheat=True)[75:150], (5, 15)))
#     state, reward, done = game.take_action(state, 5)
#     if done:
#         break
