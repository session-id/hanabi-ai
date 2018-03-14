from hanabi_sim import *
from collections import Counter

class HanabiExpert(object):
    def __init__(self, game):
        self.game = game

    def get_action(self, state):
        remaining_counter = Counter()
        for card in state.deck.cards() + [x.card for x in state.player_hands[state.cur_player]]:
            remaining_counter[card] += 1
        all_remaining = set()
        for card in state.deck.cards() + [x.card for hand in state.player_hands for x in hand]:
            all_remaining.add(card)

        game = self.game
        valid_actions = game.get_valid_actions(state)
        features = game.get_state_vector(state)

        # Check over own cards
        playable_idx = set()
        dead_idx = set()
        indispensable_idx = set()
        for card_idx, card in enumerate(state.player_hands[state.cur_player]):
            if card is None:
                raise RuntimeError("Attempting to play with less than the maximum number of cards.")
            number_vector = [1 if num in card.possible_numbers else 0 for num in range(1, game.max_number+1)]
            color_vector = [1 if color in card.possible_colors else 0 for color in game.colors]

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
            if playable == 1:
                playable_idx.add(card_idx)
            if indispensable == 1:
                indispensable_idx.add(card_idx)
            if dead_idx == 1:
                dead_idx.add(card_idx)

        if len(playable_idx) > 0:
            action = list(playable_idx)[0]
            return action

        # Pick hint
        if state.hint_tokens > 0:
            tier1_hints = set()
            tier2_hints = set()
            unhinted_numbers = []
            for other_player_num in range(1, game.num_players):
                player_id = (state.cur_player + other_player_num) % game.num_players
                for hinted_card in state.player_hands[player_id]:
                    if hinted_card.number == state.played_numbers[hinted_card.color] + 1:
                        # Playable
                        if hinted_card.number_hint and not hinted_card.color_hint:
                            tier1_hints.add(('color', other_player_num-1, hinted_card.color))
                        if hinted_card.color_hint and not hinted_card.number_hint:
                            tier1_hints.add(('number', other_player_num-1, hinted_card.number))
                        if not hinted_card.number_hint and not hinted_card.number_hint:
                            tier2_hints.add(('number', other_player_num-1, hinted_card.number))
            print("Tier 1:", tier1_hints)
            print("Tier 2:", tier2_hints)
            if len(tier1_hints) > 0:
                hint_type, person, hint = list(tier1_hints)[0]
                if hint_type == 'color':
                    print
