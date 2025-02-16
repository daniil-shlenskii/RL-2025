import time

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class Simple1DPathEnv(gym.Env):
    def __init__(self, x_start=0, x_end=10, obstacles=None, screen_width=600, screen_height=200, render_mode=None):
        super(Simple1DPathEnv, self).__init__()

        # Initialize starting and ending positions
        self.x_start = x_start
        self.x_end = x_end
        self.current_position = x_start
        
        # If no obstacles are provided, generate random ones
        self.obstacles = obstacles if obstacles else self._generate_random_obstacles()
        
        # Screen size for Pygame rendering
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.path_length = x_end - x_start

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("1D Path Environment")

        # Action space: 0 = move left, 1 = move right, 2 = jump
        self.action_space = spaces.Discrete(3)
        
        # Observation space: current position on the 1D path (continuous)
        self.observation_space = spaces.Discrete(x_end + 1)

        self.render_mode = render_mode
        
    def _generate_random_obstacles(self):
        # Randomly generate obstacles between 1 and x_end-1, avoid starting or ending point
        num_obstacles = 5
        obstacles = set(np.linspace(1, self.x_end, num_obstacles).astype(int))
        obstacles.discard(self.x_start)
        obstacles.discard(self.x_end)
        return obstacles

    def reset(self, *args, **kwargs):
        # Reset environment to start position
        self.current_position = self.x_start
        if self.render_mode == "human":
            self.render()
        return np.array([self.current_position], dtype=np.int32), {}

    def step(self, action):
        # Apply action: 0 = move left, 1 = move right, 2 = jump
        if action == 0:  # Move left
            self.current_position = max(self.x_start, self.current_position - 1)
        elif action == 1:  # Move right
            self.current_position = min(self.x_end, self.current_position + 1)
        elif action == 2:  # Jump (for simplicity, jump 2 positions ahead)
            self.current_position = min(self.x_end, self.current_position + 2)

        # Check if current position hits an obstacle
        if self.current_position in self.obstacles:
            reward = -self.x_end  # Negative reward for hitting an obstacle
            done = True 
        elif self.current_position == self.x_end:
            reward = self.x_end # Positive reward for reaching the goal
            done = True
        else:
            reward = -1  # Small negative reward to encourage faster completion
            done = False
        
        if action == 2:
            reward -= 2
        
        if self.render_mode == "human":
            self.render()

        # Return the state (current position), reward, done, and additional info
        return self.current_position, reward, done, False, {}

    def render(self):
        # Fill background with white color
        self.screen.fill((255, 255, 255))

        # Draw the path (horizontal line from start to end)
        path_color = (0, 0, 0)
        pygame.draw.line(self.screen, path_color, (50, self.screen_height // 2), 
                         (self.screen_width - 50, self.screen_height // 2), 5)

        # Draw obstacles as red blocks
        for obstacle in self.obstacles:
            obstacle_x = 50 + (self.screen_width - 100) * (obstacle - self.x_start) / self.path_length
            pygame.draw.rect(self.screen, (255, 0, 0), 
                             (obstacle_x, self.screen_height // 2 - 10, 10, 20))

        # Draw the agent (current position) as a green block
        agent_x = 50 + (self.screen_width - 100) * (self.current_position - self.x_start) / self.path_length
        pygame.draw.rect(self.screen, (0, 255, 0), 
                         (agent_x, self.screen_height // 2 - 20, 10, 20))

        # Draw the goal (end point) as a blue rectangle
        goal_x = 50 + (self.screen_width - 100) * (self.x_end - self.x_start) / self.path_length
        pygame.draw.rect(self.screen, (0, 0, 255), 
                         (goal_x, self.screen_height // 2 - 20, 10, 20))

        # Stroking effect - show possible next points
        # Move Left
        move_left_x = 50 + (self.screen_width - 100) * (self.current_position - 1 - self.x_start) / self.path_length
        if self.current_position > self.x_start and (self.current_position - 1) not in self.obstacles:
            pygame.draw.circle(self.screen, (255, 255, 0), (move_left_x, self.screen_height // 2), 8, 2)

        # Move Right
        move_right_x = 50 + (self.screen_width - 100) * (self.current_position + 1 - self.x_start) / self.path_length
        if self.current_position < self.x_end and (self.current_position + 1) not in self.obstacles:
            pygame.draw.circle(self.screen, (0, 0, 255), (move_right_x, self.screen_height // 2), 8, 2)

        # Jump (2 steps ahead)
        jump_x = 50 + (self.screen_width - 100) * (self.current_position + 2 - self.x_start) / self.path_length
        if self.current_position + 2 <= self.x_end and (self.current_position + 2) not in self.obstacles:
            pygame.draw.circle(self.screen, (0, 255, 255), (jump_x, self.screen_height // 2), 8, 2)

        # Display text: Current position
        font = pygame.font.Font(None, 36)
        text = font.render(f"Position: {self.current_position}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        # Update the screen
        pygame.display.flip()
        time.sleep(0.5)

    def close(self):
        pygame.quit()

# Example usage with Pygame rendering
if __name__ == "__main__":
    import time
    env = Simple1DPathEnv(x_start=0, x_end=20)
    env.reset()

    done = False
    total_reward = 0 
    while not done:
        env.render()
        
        time.sleep(0.5)
        # Handle Pygame events (to quit the game properly)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = env.action_space.sample()  # Random action for demonstration
        state, reward, done, _, info = env.step(action)
        print(f"Action taken: {action}, New state: {state}, Reward: {reward}")
        total_reward += reward

    print(f"Total reward: {total_reward}")

    env.close()
