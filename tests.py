from cards.cards import *
from game_state import GameState
from agent import Agent
from netty_1 import Netty
from timeit import timeit
import numpy as np
from util import one_hot

class BelchAIrtestClass():
    deck_dict = {
        Taiga(): (1, 0),
        ElvishSpiritGuide(): (4, 0),
        SimianSpiritGuide(): (4, 0),
        TinderWall(): (4, 0),
        DesperateRitual(): (4, 0),
        PyreticRitual(): (3, 0),
        RiteOfFlame(): (4, 0),
        SeethingSong(): (4, 0),
        EmptyTheWarrens(): (3, 1),
        GitaxianProbe(): (4, 0),
        LandGrant(): (4, 0),
        LionsEyeDiamond(): (4, 0),
        LotusPetal(): (4, 0),
        Manamorphose(): (2, 0),
        GoblinCharbelcher(): (4, 0),
        ChromeMox(): (3, 0),
        BurningWish(): (4, 0),
        ReforgeTheSoul(): (0, 1)
    }

    def setup_test_gamestate(self):
        self.test_game_state = GameState()
        self.test_game_state.add_deck(self.deck_dict)

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
        self.test_game_state.reset_game(False)
        print('Successfully reset game')
        assert True

    def testGame(self):
        self.game_state.reset_game()
        self.game_state.all_actions()
        print('Sucessfully ran all actions')
        assert True