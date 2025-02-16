import random
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

warnings.filterwarnings("ignore", category=DeprecationWarning)

class CardEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        super(CardEnv, self).__init__(*args, **kwargs)
        self.current_step = 0

        # Action: Open next card (2) Guess, that next card is black (0) Guess, that next card is red (1)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(27 * 27)
        self.state = 27 * 27 - 1
        self.deck = [0] * 26 + [1] * 26
        np.random.shuffle(self.deck)

    def reset(self, seed: int, *args, **kwargs):
        np.random.seed(seed)
        self.current_step = 0
        self.deck = [0] * 26 + [1] * 26
        np.random.shuffle(self.deck)
        self.state = 27 * 27 - 1
        return self.state, {}

    def step(self, action):
        self.current_step +=1
        card = self.deck[0]
        self.deck = self.deck[1:]
        self.state -= 27 ** card
        reward = 0
        done = False
        #If we just open next card
        if action == 2:
            if self.current_step == 26:
                done = True
        #If we try to guess the card
        else:
            done = True
            reward = (int(action == card) - 0.5) * 2 #1, if correct; -1 if incorrect

        return self.state, reward, done, False, {}

    def render(self):
        print("There are ", self.state % 27, "black cards and", self.state // 27, "red cards in the deck")
