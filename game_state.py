from color_dict import ColorDict
from config import COLORS
from action import Action
from consequences import UntapPermanents, StormCountZero, DealGoblinDamage, \
                         DrawCard, ResetManaPool, AddTurn
from numpy.random import choice
from random import shuffle


class GameState(object):
    def __init__(self):
        self.deck = {}
        self.hand = {}
        self.exile = {}
        self.sideboard = {}
        self.battlefield = {}
        self.graveyard = {}
        self.tapped = {}

        self.mana_pool = ColorDict()
        self.goblins = 0
        self.storm_count = 0
        self.opp_life_total = 20
        self.taiga_bottom = False
        self.turn = 1
        self.lost = False
        self.won = False

        self.actions_set = False
        self.actions = []

        self.cards = []

    def add_card(self, card, maindeck, sideboard):
        self.cards.append((card, maindeck, sideboard))

    def add_deck(self, deck):
        deck_list = [self.add_card(card, main_cnt, sb_cnt) for card, (main_cnt, sb_cnt) in deck.items()]
        print('added {} cards to maindeck'.format(len(deck_list)))
        return deck_list

    def reset_game(self, draw_opening_hand=True, test=False):
        for card, maindeck, sideboard in self.cards:
            self.deck[card.name] = maindeck
            if sideboard > 0:
                self.sideboard[card.name] = sideboard
            self.hand[card.name] = 0
            self.exile[card.name] = 0
            if card.is_permanent:
                self.battlefield[card.name] = 0
            self.graveyard[card.name] = 0
            if card.is_tappable:
                self.tapped[card.name] = 0
        self.mana_pool = ColorDict()
        self.goblins = 0
        self.storm_count = 0
        self.opp_life_total = 20
        self.taiga_bottom = False
        self.turn = 1
        self.lost = False
        self.won = False
        if draw_opening_hand:
            self.draw_opening_hand()
        if test:
            self.add_mana(ColorDict({'Red': 2}))
            self.increase_card_count('Goblin Charbelcher', 'Battlefield')
            self.increase_card_count('Taiga', 'Battlefield')
            #self.increase_card_count('Seething Song', 'Hand')
            self.reduce_card_count('Goblin Charbelcher', 'Deck')
            self.reduce_card_count('Taiga', 'Deck')
            #self.reduce_card_count('Seething Song', 'Deck')

    def possible_actions(self):
        """
        List of binary numbers to represent if that action is available

        :return:
        """
        all_actions = self.all_actions()
        legal_actions = [x.allowed(self) * 1 for x in all_actions]
        return legal_actions, all_actions

    def reward(self):
        if not self.won and not self.lost:
            return None
        if self.won:
            return 60 - self.turn
        return 1

    def all_actions(self):
        if self.actions_set:
            return self.actions
        self.actions = [Action('Game',
                               'Pass turn',
                               requirements=[],
                               consequences=[AddTurn(),
                                             UntapPermanents(),
                                             ResetManaPool(),
                                             StormCountZero(),
                                             DrawCard(),
                                             DealGoblinDamage()
                                             ])]
        for card, _, _ in self.cards:
            print(card.name, len(self.actions))
            self.actions.extend(card.actions)
        self.actions_set = True
        return self.actions

    def state_space(self):
        # We should make it so we can just add em all up for the correct representation.
        # The objects should know
        # Can we cache this data per episode?
        # we can provide ways to invalidate cache whenever deck size changes

        representation_list = []
        for card, _, _ in self.cards:
            representation_list.append(self.deck[card.name])
        for card, _, _ in self.cards:
            representation_list.append(self.hand[card.name])
        for card, _, _ in self.cards:
            representation_list.append(self.battlefield[card.name])
        for card, _, _ in self.cards:
            representation_list.append(self.graveyard[card.name])
        for card, _, _ in self.cards:
            if card.name in self.tapped:
                representation_list.append(self.tapped[card.name])
        for card, _, _ in self.cards:
            if card.name in self.sideboard:
                representation_list.append(self.sideboard[card.name])

        for color in COLORS:
            representation_list.append(self.mana_pool[color])
        representation_list.append(self.goblins)
        representation_list.append(self.storm_count)
        representation_list.append(self.opp_life_total)
        representation_list.append(self.taiga_bottom * 1)
        representation_list.append(self.turn)
        number_cards_in_deck = sum([self.deck[k] for k in self.deck])
        representation_list.append(number_cards_in_deck)
        return representation_list

    def untapped(self, card):
        return self.battlefield[card] > self.tapped[card]

    def mana_floating(self, c_dict):
        for k in c_dict:
            if self.mana_pool[k] < c_dict[k]:
                return False
        return True

    def card_in_hand(self, card):
        return self.hand[card] > 0

    def card_in_play(self, card):
        return self.battlefield[card] > 0

    def card_in_sideboard(self, card):
        return self.sideboard[card] > 0

    def card_in_deck(self, card):
        return self.deck[card] > 0

    def storm_count_zero(self):
        self.storm_count = 0

    def goblin_damage(self):
        self.opp_life_total -= self.goblins
        if self.opp_life_total < 1:
            self.won = True

    def damage_opponent(self, damage):
        self.opp_life_total -= damage
        if self.opp_life_total < 1:
            self.won = True

    def untap_permanents(self):
        for card, _, _ in self.cards:
            if card.is_tappable:
                self.tapped[card.name] = 0

    def add_rite_mana(self):
        self.add_mana(ColorDict({'Red': 2 + self.graveyard['Rite of Flame']}))

    def belch(self):
        deck_list = self._list_deck(True, True)
        total_damage = 0
        for card in deck_list:
            if card == 'Taiga':
                total_damage *= 2
                break
            total_damage += 1
        self.damage_opponent(total_damage)
        self.taiga_bottom = True

    def discard_hand(self):
        for card_name in self.hand:
            self.graveyard[card_name] += self.hand[card_name]
            self.hand[card_name] = 0

    def shuffle(self):
        self.taiga_bottom = False

    def draw_cards(self, amount):
        list_deck = self._list_deck()
        if len(list_deck) < amount:
            self.lost = True
        else:
            cards_to_draw = choice(list_deck, amount, False)
            for card_to_draw in cards_to_draw:
                self.deck[card_to_draw] -= 1
                self.hand[card_to_draw] += 1

    def _list_deck(self, include_Taiga=True, shuffle_list=False):
        deck_list = []
        for card_name in self.deck:
            if card_name != 'Taiga' or not self.taiga_bottom:
                deck_list.extend([card_name for _ in range(self.deck[card_name])])
        if shuffle_list:
            shuffle(deck_list)
        if self.taiga_bottom and include_Taiga and self.deck['Taiga'] > 0:
            deck_list.append('Taiga')
        return deck_list

    def tap_card(self, card):
        self.tapped[card] += 1

    def add_storm(self):
        self.storm_count += 1

    def add_goblins(self):
        self.goblins += 2 * self.storm_count

    def add_mana(self, c_dict):
        self.mana_pool.add_mana(c_dict)

    def reduce_mana(self, c_dict):
        self.mana_pool.subtract_mana(c_dict)

    def reset_mana_pool(self):
        self.mana_pool.subtract_mana(self.mana_pool)

    def increase_card_count(self, card, zone):
        self._zone_dispatcher(zone)[card] += 1

    def reduce_card_count(self, card, zone):
        self._zone_dispatcher(zone)[card] -= 1

    def draw_opening_hand(self):
        self.draw_cards(7)

    def add_turn(self):
        self.turn += 1

    def _zone_dispatcher(self, zone):
        if zone == 'Graveyard':
            return self.graveyard
        if zone == 'Battlefield':
            return self.battlefield
        elif zone == 'Hand':
            return self.hand
        elif zone == 'Exile':
            return self.exile
        elif zone == 'Deck':
            return self.deck
        elif zone == 'Sideboard':
            return self.sideboard
        raise ValueError('Incorrect zone type', zone)

    def __str__(self):
        # It'd be nice to rely on something higher level to represent the game in string form.
        # I find big functions like this hard to manage, but they might not ever really change.

        repr_str = 'GAME STATE: \n'
        #repr_str += '  Cards in hand:      ' + str(sum([self.hand[k] for k in self.hand])) + '\n'
        #repr_str += '  Cards in deck:      ' + str(sum([self.deck[k] for k in self.deck])) + '\n'
        #repr_str += '  Cards in play:      ' + str(sum([self.battlefield[k] for k in self.battlefield])) + '\n'
        #repr_str += '  Cards in graveyard: ' + str(sum([self.graveyard[k] for k in self.graveyard])) + '\n'
        #repr_str += '  Cards in sideboard: ' + str(sum([self.sideboard[k] for k in self.sideboard])) + '\n'


        repr_str += 'Hand: ' + str(sum([self.hand[k] for k in self.hand])) + ', '
        repr_str += 'Deck: ' + str(sum([self.deck[k] for k in self.deck])) + ', '
        repr_str += 'Play: ' + str(sum([self.battlefield[k] for k in self.battlefield])) + ', '
        repr_str += 'GY: ' + str(sum([self.graveyard[k] for k in self.graveyard])) + ', '
        #repr_str += '  Cards in sideboard: ' + str(sum([self.sideboard[k] for k in self.sideboard])) + '\n'

        repr_str += '\nHAND: \n'
        for k in self.hand:
            if self.hand[k] > 0:
                repr_str += k + ': ' + str(self.hand[k]) + ', '
        repr_str += '\nPLAY: \n'
        for k in self.battlefield:
            if self.battlefield[k] > 0:
                repr_str += k + ': ' + str(self.battlefield[k]) + ', '
        repr_str += '\nTAPPED: \n'
        for k in self.tapped:
            if self.tapped[k] > 0:
                repr_str += k + ': ' + str(self.tapped[k]) + ', '

        repr_str += '\nMANA POOL: '
        for c in COLORS:
            repr_str += '  ' + c + ': ' + str(self.mana_pool[c]) + ', '
        repr_str += '\nACTIONS: ' + str(sum(self.possible_actions()[0])) + '\n'
        repr_str += 'GOBLINS: ' + str(self.goblins) + '\n'
        repr_str += 'TURN: ' + str(self.turn) + '\n'
        repr_str += 'OPP LIFETOTAL: ' + str(self.opp_life_total) + '\n'
        return repr_str
