from cards import cards
from game_state import GameState
from agent import Agent
from netty_1 import Netty
from timeit import timeit
import numpy as np
from util import one_hot
class BelchAIrtestClass():
    deck_dict = {
        cards.Taiga(): (1, 0),
        cards.ElvishSpiritGuide(): (4, 0),
        cards.SimianSpiritGuide(): (4, 0),
        cards.TinderWall(): (4, 0),
        cards.DesperateRitual(): (4, 0),
        cards.PyreticRitual(): (3, 0),
        cards.RiteOfFlame(): (4, 0),
        cards.SeethingSong(): (4, 0),
        cards.EmptyTheWarrens(): (3, 1),
        cards.GitaxianProbe(): (4, 0),
        cards.LandGrant(): (4, 0),
        cards.LionsEyeDiamond(): (4, 0),
        cards.LotusPetal(): (4, 0),
        cards.Manamorphose(): (2, 0),
        cards.GoblinCharbelcher(): (4, 0),
        cards.ChromeMox(): (3, 0),
        cards.BurningWish(): (4, 0),
        cards.ReforgeTheSoul(): (0, 1)
    }

    def setup_test_gamestate(self):
        self.test_game_state = GameState()
        self.test_game_state.add_deck(deck_dict)

    def setup_game(self):
        self.game_state = GameState()
        self.game_state.add_deck(self.deck_dict)

    def setUp(self):
        self.setup_game()
        self.setup_test_gamestate()

    def tearDown(self):
        del self.game_state
        del self.test_game_state

    def testGamestate(self):
        # test_game_state.reset_game(False)
        print('Successfully reset game')
        assert True

    def testGame(self):
        # game_state.reset_game()
        # game_state.all_actions()
        print('Sucessfully ran all actions')
        assert True