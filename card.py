from mana_iterator import color_combinations, fill_up_remaining_colors
from requirements import PayMana, InHand, InPlay, Untapped
from consequences import ReduceMana, AddMana, MoveCard, Tap, AddStorm
from color_dict import ColorDict
from action import Action


class Card(object):
    def __init__(self, name):
        self.name = name
        self.actions = []

    def add_action(self, action):
        self.actions.append(action)

    def add_mana_action(self, action, paying=None, adding=None):
        if isinstance(paying, ColorDict):
            paying = [paying]
        if isinstance(adding, ColorDict):
            adding = [adding]
        if paying is not None and adding is None:
            for option in paying:
                next_action = action.copy()
                next_action.add_requirement(PayMana(option))
                next_action.add_consequence(ReduceMana(option))
                self.add_action(next_action)
        elif paying is None and adding is not None:
            for option in adding:
                next_action = action.copy()
                next_action.add_consequence(AddMana(option))
                self.add_action(next_action)
        elif paying is not None and adding is not None:
            for option_add in adding:
                for option_pay in paying:
                    next_action = action.copy()
                    next_action.add_requirement(PayMana(option_pay))
                    next_action.add_consequence(ReduceMana(option_pay))
                    next_action.add_consequence(AddMana(option_add))
                    self.add_action(next_action)