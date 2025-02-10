from typing import Optional

import gymnasium as gym
import numpy as np
import pygame


class PointEnv(gym.Env):
    x_start: float = 0.
    screen_dim: int = 500

    def __init__(self, size: float=10., render_mode: Optional[str]=None):
        self.size = size

        self.observation_space = gym.spaces.Box(-1., 1., shape=(1,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,))

        self._x_location = self.x_start
        self._terminated = False

        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.isopen = True

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)

        x_coord = self.x_start + (np.random.rand() - 0.5) * 2. / self.size

        observation = np.asarray([x_coord])
        info = {}

        self._x_location = x_coord

        if self.render_mode is not None:
            self.render()

        return observation, info

    def step(self, action: np.float32):
        step = action[0] / self.size

        new_x_coord = self._x_location + step

        observation = np.asarray([new_x_coord])
        reward = step
        terminated = not (-1. < new_x_coord < 1.)
        truncated = False
        info = {}

        self._x_location = new_x_coord
        self._terminated = terminated

        if self.render_mode is not None:
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            pygame.display.set_mode((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # Clear the screen with white color

        # Scale the x location to fit in the screen
        x_pos = int((self._x_location + 1) * (self.screen_dim // 2))
        y_pos = self.screen_dim // 2

        # Draw the point as a red circle
        pygame.draw.circle(self.screen, (255, 0, 0), (x_pos, y_pos), 10)

        pygame.display.flip()  # Update the display
        self.clock.tick(30)  # Limit the frame rate to 60 FPS

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
