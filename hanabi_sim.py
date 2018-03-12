from __future__ import print_function

from collections import Counter
import copy
import numpy as np
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

    def __lt__(self, other):
        return (self.color, self.number) < (other.color, other.number)

    def __eq__(self, other):
        return self.number == other.number and self.color == other.color

class Deck(object):
    '''A stack of Hanabi cards.'''

    def __init__(self, cards_and_counts):
        '''
        Args
        - cards_and_counts: dict, {Card => int}
        '''
        self.cards_set = set()
        for card, count in cards_and_counts.items():
            for i in range(count):
                self.cards_set.add((card, i))

    def cards(self):
        '''
        Returns: list of Card
        '''
        return [card for card, _ in self.cards_set]

    def draw(self):
        '''
        Returns: Card
        '''
        drawn_card = random.sample(self.cards_set, 1)[0]
        self.cards_set.remove(drawn_card)
        return drawn_card[0]

    def __len__(self):
        return len(self.cards_set)

class BaseHanabiGame(object):
    def __init__(self, num_players):
        raise NotImplementedError

    def get_start_state(self):
        '''
        Returns
            start_state: a randomized starting state for the game.
        '''
        raise NotImplementedError

    def get_state_vector_size(self):
        raise NotImplementedError

    def get_num_actions(self):
        raise NotImplementedError

    def get_state_vector(self, state):
        '''
        Returns
            state_vector: a vector containing the feature vector corresponding to the game state
                as seen by the provided player
        '''
        raise NotImplementedError

    def get_valid_actions(self, state):
        '''
        Returns
            valid_actions: a set containing the indices of valid actions
        '''
        raise NotImplementedError

    def take_action(self, state, action):
        '''
        Args
            state: current_state
            action: int, corresponding to index of action taken. actions are numbered in
                blocks as follows. x = # of cards in hand, n = # of players, h = # of possible hints
                0 - x-1: play card
                x - 2x-1: discard card
                2x - 2x+h-1: hints for other player 1
                2x+h - 2x+2h-1: hints for other player 2
                ...
                2x+(n-2)h - 2x+(n-1)h-1: hints for player n-1

        Returns
            new_state: the new global state
            reward: the reward from performing action
            done: a boolean indicating if the episode has finished
        '''
        raise NotImplementedError
        return state, reward, done

class HintedCard(object):
    def __init__(self, card, max_number, colors):
        '''
        Args
        - card: Card
        - max_number: int
        - colors: list of str
        '''
        self.card = card
        self.color_hint = False
        self.number_hint = False
        self.possible_numbers = set(list(range(1, max_number+1)))
        self.possible_colors = set(colors)

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
        '''
        Args
        - num_players: int
        - cards_per_player: int
        - colors: list of str
        - max_number: int, highest number per color
        - number_counts: int
        - max_hint_tokens: int
        - starting_bomb_tokens: int
        '''
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

        self.history = []

    ''' This method is just for printing. '''
    def get_pretty_board(self):
        return [Card(num, color) for color, num in self.played_numbers.items()]

    def initialize_random(self):
        '''
        Initialize a random starting hand for all players.
        '''
        self.deck = copy.deepcopy(self.base_deck)
        self.player_hands = []
        for player_num in range(self.num_players):
            hand = []
            for i in range(self.cards_per_player):
                hand.append(HintedCard(self.deck.draw(), self.max_number, self.colors))
            self.player_hands.append(hand)
        self.cur_player = 0
        self.played_numbers = {color: 0 for color in self.colors}
        self.hint_tokens = self.max_hint_tokens
        self.bomb_tokens = self.starting_bomb_tokens
        self.turns_until_end = self.num_players

    def add_hint_token(self):
        self.hint_tokens = min(self.max_hint_tokens, self.hint_tokens + 1)

    def advance_player(self):
        self.cur_player = (self.cur_player + 1) % self.num_players

    def print_self(self):
        print("Player {}, Hint {}, Bomb {}, Cards left {}".format(self.cur_player, self.hint_tokens, self.bomb_tokens, len(self.deck)))
        print("\t{ " + ", ".join(colored(self.played_numbers[color], COLOR_TO_TERM[color]) for color in self.colors) + " }")
        print("\t{}".format(self.player_hands))

class RegularHanabiGameEasyFeatures(object):
    '''
    The regular Hanabi game, 5 colors.
    Currently returning the easy feature set
    '''

    def __init__(self, num_players, colors, cards_per_player, max_number, number_counts):
        self.num_players = num_players
        self.cards_per_player = cards_per_player
        self.colors = colors
        self.num_colors = len(self.colors)
        self.max_number = max_number
        self.max_hint_tokens = 8
        self.starting_bomb_tokens = 3
        self.number_counts = number_counts

    def get_start_state(self):
        '''
        Returns
        - state: HanabiState
        '''
        state = HanabiState(self.num_players, self.cards_per_player, self.colors, self.max_number,
            self.number_counts, self.max_hint_tokens, self.starting_bomb_tokens)
        state.initialize_random()
        return state

    def get_action_names(self, state):
        '''
        Returns
        - action_names: list of str, descriptions of each action
        '''
        action_names = []
        # Play own cards
        for i in range(self.cards_per_player):
            action_names.append("Play slot {}".format(i))
        # Discard own cards
        for i in range(self.cards_per_player):
            action_names.append("Discard slot {}".format(i))
        # Give hints
        for other_player_num in range(1, self.num_players):
            player_id = (state.cur_player + other_player_num) % self.num_players
            for number in range(1, self.max_number+1):
                action_names.append("Player {}: hint {}'s".format(player_id, number))
            for color in self.colors:
                action_names.append("Player {}: hint {}"\
                    .format(player_id, colored(color, COLOR_TO_TERM[color])))
        return action_names

    def get_valid_actions(self, state):
        '''
        Returns
        - valid_actions: set of int
        '''
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
        if state.hint_tokens > 0:
            for other_player_num in range(1, self.num_players):
                player_id = (state.cur_player + other_player_num) % self.num_players
                player_hand = state.player_hands[player_id]
                for number in range(1, self.max_number+1):
                    if any(card.number == number for card in player_hand):
                        if not all(card.number_hint for card in player_hand if card.number == number):
                            valid_actions.add(action_num)
                    action_num += 1
                for color in self.colors:
                    if any(card.color == color for card in player_hand):
                        if not all(card.color_hint for card in player_hand if card.color == color):
                            valid_actions.add(action_num)
                    action_num += 1
        return valid_actions

    def get_num_actions(self):
        return 2 * self.cards_per_player + (self.num_players - 1) * (self.num_colors + self.max_number)

    def take_action(self, state, action, bomb_reward=0., alive_reward=0.):
        '''
        Args
        - state: HanabiState
        - action: int

        Returns: (state, reward, done)
        - state: HanabiState
        - reward: float
        - done: bool
        '''
        done = False
        reward = 0

        # If not needed, we can remove this copying for more efficiency
        state = copy.deepcopy(state)
        if action not in self.get_valid_actions(state):
            raise RuntimeError("Invalid action: {}".format(action))

        # Play own card
        if action < self.cards_per_player:
            played_index = action
            card_played = state.player_hands[state.cur_player][played_index].card
            if state.played_numbers[card_played.color] + 1 == card_played.number:
                if card_played.number == 5:
                    state.add_hint_token()
                state.played_numbers[card_played.color] += 1
                reward += 1
            else:
                state.bomb_tokens -= 1
                if len(state.deck) + state.turns_until_end > state.bomb_tokens:
                    reward += bomb_reward
        # Discard
        elif action < self.cards_per_player * 2:
            played_index = action - self.cards_per_player
            state.add_hint_token()

        # For both Discard and Play, need to redraw and shift
        if action < self.cards_per_player * 2:
            if len(state.deck) > 0:
                drawn_card = HintedCard(state.deck.draw(), self.max_number, self.colors)
                previous_hand = state.player_hands[state.cur_player]
                state.player_hands[state.cur_player] = [drawn_card] + previous_hand[:played_index]\
                    + previous_hand[played_index+1:]
            else:
                previous_hand = state.player_hands[state.cur_player]
                state.player_hands[state.cur_player] = previous_hand[:played_index]\
                    + previous_hand[played_index+1:]

        # Hint
        if action >= self.cards_per_player * 2:
            state.hint_tokens -= 1
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
                        card.possible_numbers = set([card.number])
                    elif hinted_number in card.possible_numbers:
                        card.possible_numbers.remove(hinted_number)
            # Color hint
            else:
                hinted_color = self.colors[t - self.max_number]
                for card in state.player_hands[player_hinted]:
                    if card.color == hinted_color:
                        card.color_hint = True
                        card.possible_colors = set([card.color])
                    elif hinted_color in card.possible_colors:
                        card.possible_colors.remove(hinted_color)

        if state.turns_until_end <= 0:
            done = True

        if len(state.deck) == 0:
            state.turns_until_end -= 1

        # If perfect score achieved
        if all(num == self.max_number for num in state.played_numbers.values()):
            done = True

        if state.bomb_tokens == 0:
            done = True

        state.advance_player()

        if not done:
            reward += alive_reward

        return state, reward, done

    def get_state_vector(self, state, cheat=False):
        if cheat:
            for player_hand in state.player_hands:
                for hinted_card in player_hand:
                    hinted_card.color_hint = True
                    hinted_card.number_hint = True
                    hinted_card.possible_numbers = set([hinted_card.number])
                    hinted_card.possible_colors = set([hinted_card.color])

        remaining_counter = Counter()
        for card in state.deck.cards() + [x.card for x in state.player_hands[state.cur_player]]:
            remaining_counter[card] += 1
        all_remaining = set()
        for card in state.deck.cards() + [x.card for hand in state.player_hands for x in hand]:
            all_remaining.add(card)

        all_vectors = []

        # Other players' cards
        for other_player_num in range(1, self.num_players):
            player_id = (state.cur_player + other_player_num) % self.num_players
            # TODO: the one hot encoding for number + color + None has one degree of linear redundancy
            if len(state.player_hands[player_id]) == self.cards_per_player - 1:
                    all_vectors.append([0] * (self.num_colors + self.max_number + 5))
            for hinted_card in state.player_hands[player_id]:
                if hinted_card is not None:
                    number_vector = [0] * self.max_number
                    number_vector[hinted_card.number - 1] = 1
                    color_vector = [0] * self.num_colors
                    color_vector[self.colors.index(hinted_card.color)] = 1

                    playable = dead = indispensable = 0
                    if hinted_card.number == state.played_numbers[hinted_card.color] + 1:
                        playable = 1
                    if hinted_card.number <= state.played_numbers[hinted_card.color]:
                        dead = 1
                    else:
                        for num in range(state.played_numbers[hinted_card.color]+1, hinted_card.number):
                            if Card(num, hinted_card.color) not in all_remaining:
                                dead = 1
                    if dead != 1 and remaining_counter[hinted_card.card] == 0:
                        active_copies = 0
                        for player_num in range(self.num_players):
                            for hinted_card2 in state.player_hands[player_num]:
                                if hinted_card2.card == hinted_card.card:
                                    active_copies += 1
                        if active_copies == 1:
                            indispensable = 1
                    extras_vector = [int(hinted_card.number_hint), int(hinted_card.color_hint), playable, dead, indispensable]

                    card_vector = number_vector + color_vector + extras_vector

                all_vectors.append(card_vector)

        if len(state.player_hands[state.cur_player]) == self.cards_per_player - 1:
            all_vectors.append([0] * (self.num_colors + self.max_number + 5))
        # Your own cards
        for card in state.player_hands[state.cur_player]:
            if card is None:
                card_vector = [0] * (self.num_colors + self.max_number + 5)
            else:
                number_vector = [1 if num in card.possible_numbers else 0 for num in range(1, self.max_number+1)]
                color_vector = [1 if color in card.possible_colors else 0 for color in self.colors]

                total_possibilities = 0
                indispensable_poss = 0
                playable_poss = 0
                dead_poss = 0
                for possible_card in remaining_counter:
                    if remaining_counter[possible_card] > 0:
                        if possible_card.number in card.possible_numbers and possible_card.color in card.possible_colors:
                            total_possibilities += 1
                            if state.played_numbers[possible_card.color] + 1 == possible_card.number:
                                playable_poss += 1
                            dead = False
                            # Check if dead
                            if state.played_numbers[possible_card.color] >= possible_card.number:
                                dead_poss += 1
                                dead = True
                            else:
                                for num in range(state.played_numbers[possible_card.color]+1, possible_card.number):
                                    if Card(num, possible_card.color) not in all_remaining:
                                        dead_poss += 1
                                        dead = True
                                        break
                            # Note: indispensable is mutually exclusive with dead
                            if not dead and remaining_counter[possible_card] == 1:
                                indispensable_poss += 1
                if total_possibilities == 0:
                    raise RuntimeError("There are zero possibilities for what this card can be. Something went wrong.")
                playable = int(playable_poss == total_possibilities)
                indispensable = int(indispensable_poss == total_possibilities)
                dead = int(dead_poss == total_possibilities)
                extras_vector = [int(card.number_hint), int(card.color_hint), playable, dead, indispensable]

                card_vector = number_vector + color_vector + extras_vector

            all_vectors.append(card_vector)

        # Hint token
        hint_vector = [0] * self.max_hint_tokens
        if state.hint_tokens > 0:
            hint_vector[state.hint_tokens - 1] = 1
        bomb_vector = [0] * (self.starting_bomb_tokens - 1)
        if state.bomb_tokens > 1:
            bomb_vector[state.bomb_tokens - 2] = 1
        all_vectors.append(hint_vector)
        all_vectors.append(bomb_vector)

        # Already played cards
        for color in self.colors:
            x = [0] * self.max_number
            if state.played_numbers[color] > 0:
                x[state.played_numbers[color] - 1] = 1
            all_vectors.append(x)

        # Deck exhausted
        exhausted = [0]
        if len(state.deck) == 0:
            exhausted = [1]
        all_vectors.append(exhausted)

        feature_vector = np.array([item for sublist in all_vectors for item in sublist])
        if len(feature_vector) != self.get_state_vector_size():
            raise RuntimeError("Feature vector length is incorrect: {}, supposed to be {}".format(len(feature_vector), self.get_state_vector_size()))

        return feature_vector

    def get_state_vector_size(self):
        return (self.num_colors + self.max_number + 5) * self.cards_per_player * self.num_players + self.max_hint_tokens\
            + (self.starting_bomb_tokens-1) + self.max_number * self.num_colors + 1

class RegularHanabiGameHistoryFeatures(RegularHanabiGameEasyFeatures):
    '''
    The regular Hanabi game, 5 colors.
    Using history as the primary feature, ignoring actual card identities.
    '''
    def take_action(self, state, action):
        state, reward, done = super().take_action(state, action)
        x = np.zeros((self.get_num_actions()))
        x[action] = 1
        state.history.append(x)
        return state, reward, done

    def get_state_vector(self, state):
        remaining_counter = Counter()
        for card in state.deck.cards() + [x.card for x in state.player_hands[state.cur_player]]:
            remaining_counter[card] += 1
        all_remaining = set()
        for card in state.deck.cards() + [x.card for hand in state.player_hands for x in hand]:
            all_remaining.add(card)

        all_vectors = []

        # Other players' cards
        for other_player_num in range(1, self.num_players):
            player_id = (state.cur_player + other_player_num) % self.num_players
            if len(state.player_hands[player_id]) == self.cards_per_player - 1:
                    all_vectors.append([0] * 3)
            for hinted_card in state.player_hands[player_id]:
                if hinted_card is not None:
                    playable = dead = indispensable = 0
                    if hinted_card.number == state.played_numbers[hinted_card.color] + 1:
                        playable = 1
                    if hinted_card.number <= state.played_numbers[hinted_card.color]:
                        dead = 1
                    else:
                        for num in range(state.played_numbers[hinted_card.color]+1, hinted_card.number):
                            if Card(num, hinted_card.color) not in all_remaining:
                                dead = 1
                    if dead != 1 and remaining_counter[hinted_card.card] == 0:
                        active_copies = 0
                        for player_num in range(self.num_players):
                            for hinted_card2 in state.player_hands[player_num]:
                                if hinted_card2.card == hinted_card.card:
                                    active_copies += 1
                        if active_copies == 1:
                            indispensable = 1
                    card_vector = [playable, dead, indispensable]

                all_vectors.append(card_vector)

        # Hint token
        hint_vector = [0] * self.max_hint_tokens
        if state.hint_tokens > 0:
            hint_vector[state.hint_tokens - 1] = 1
        bomb_vector = [0] * (self.starting_bomb_tokens - 1)
        if state.bomb_tokens > 1:
            bomb_vector[state.bomb_tokens - 2] = 1
        all_vectors.append(hint_vector)
        all_vectors.append(bomb_vector)

        # Deck exhausted
        exhausted = [0]
        if len(state.deck) == 0:
            exhausted = [1]
        all_vectors.append(exhausted)

        static_vector = np.array([item for sublist in all_vectors for item in sublist])
        dynamic_vectors = state.history

        if len(static_vector) != self.get_state_vector_size()['static']:
            raise RuntimeError("Static vector length is incorrect: {}, supposed to be {}".format(len(static_vector), self.get_state_vector_size()['static']))

        for vec in dynamic_vectors:
            if len(vec) != self.get_state_vector_size()['dynamic']:
                raise RuntimeError("Dynamic vector length is incorrect: {}, supposed to be {}".format(len(vec), self.get_state_vector_size()['dynamic']))

        return {'static': static_vector, 'dynamic': dynamic_vectors}

    def get_state_vector_size(self):
        d = {
                'static': 3 * (self.num_players - 1) * self.cards_per_player + self.max_hint_tokens + (self.starting_bomb_tokens-1) + 1,
                'dynamic': self.get_num_actions()
            }
        return d
