import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class TRexJump(gym.Env):
    """
    T-Rex Runner environment for reinforcement learning.
    The agent controls a character that must jump over obstacles.
    
    Actions:
        0: No jump
        1: Jump
        
    Observation:
        [0]: Player y position (normalized)
        [1]: Is jumping flag (0 or 1)
        [2]: Next obstacle distance (normalized)
        [3]: Second obstacle distance (normalized)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, num_obstacles=10):
        # Screen dimensions
        self.width = 800
        self.height = 400
        self.ground_height = 300
        
        # Game elements
        self.jump_height = 100
        self.player_width, self.player_height = 40, 60
        self.player_x = 100
        self.num_obstacles = num_obstacles
        self.obstacle_width = 30
        self.obstacle_height = 50
        
        # Game parameters
        self.min_gap, self.max_gap = 250, 450
        self.speed = 10.0
        self.gravity = 1.0
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Rendering setup
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.isopen = True
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset player state
        self.player_y = self.ground_height - self.player_height
        self.player_vel_y = 0
        self.is_jumping = False
        
        # Generate new obstacles
        self.obstacles = []
        self.generate_obstacles()
        
        # Reset game state
        self.done = False
        self.truncated = False
        self.score = 0
        self.distance_traveled = 0
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}
    
    def calculate_min_gap(self):
        """Calculate minimum safe gap based on jump physics"""
        jump_time = 2 * math.sqrt(2 * self.jump_height / self.gravity)
        min_distance = self.speed * jump_time * 1.2  # Add safety margin
        
        return max(self.min_gap, int(min_distance))
    
    def generate_obstacles(self):
        """Generate a sequence of obstacles with appropriate gaps"""
        x = self.width + 100  # First obstacle position
        
        for i in range(self.num_obstacles):
            # Calculate gap
            min_gap = self.calculate_min_gap()
                
            # Adjust gap with difficulty progression
            gap = self.np_random.integers(min_gap, self.max_gap)
            difficulty_factor = min(1.0 + (i * 0.05), 1.3)
            gap = int(gap * difficulty_factor)
            
            x += gap
            
            self.obstacles.append({
                "x": x,
                "height": self.obstacle_height,
                "passed": False
            })
            
        # Set goal position
        self.goal_x = self.obstacles[-1]["x"] + 500
    
    def _get_obs(self):
        """Return the current observation state"""
        # Player state
        player_height_from_ground = self.ground_height - self.player_y - self.player_height
        player_y_norm = min(1.0, max(0.0, player_height_from_ground / self.jump_height))
        is_jumping = 1.0 if self.is_jumping else 0.0
        
        # Obstacle information
        next_obstacles = [obs for obs in self.obstacles if not obs["passed"]][:2]
        
        # Add dummy obstacles if needed
        while len(next_obstacles) < 2:
            next_obstacles.append({
                "x": self.width*2,
                "passed": False
            })
        
        obs = [player_y_norm, is_jumping]
        
        for obstacle in next_obstacles:
            true_distance = obstacle["x"] - (self.player_x + self.player_width)
            distance_norm = min(1.0, max(0.0, true_distance / self.width))
            obs.append(distance_norm)
            
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): 0 = no jump, 1 = jump
            
        Returns:
            observation, reward, done, truncated, info
        """
        # Update game state
        self.distance_traveled += self.speed
        
        # Process jump action
        if action == 1 and not self.is_jumping:
            self.is_jumping = True
            self.player_vel_y = -np.sqrt(2 * self.gravity * self.jump_height)
        
        # Update player position
        if self.is_jumping:
            self.player_y += self.player_vel_y
            self.player_vel_y += self.gravity
            
            if self.player_y >= self.ground_height - self.player_height:
                self.player_y = self.ground_height - self.player_height
                self.player_vel_y = 0
                self.is_jumping = False
        
        # Basic reward (penalize jumping)
        reward = 1 - action * 2

        # Update obstacles
        for obstacle in self.obstacles:
            obstacle["x"] -= self.speed
            
            if not obstacle["passed"] and obstacle["x"] + self.obstacle_width < self.player_x:
                obstacle["passed"] = True
                self.score += 1
                reward += 50
        
        # Check collision
        if self._check_collision():
            self.done = True
            reward -= 50
        
        # Check goal
        self.goal_x -= self.speed
        if self.goal_x <= self.player_x:
            self.done = True
            reward += 100
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, self.done, self.truncated, {"score": self.score}
    
    def _check_collision(self):
        """Check if the player collides with any obstacle"""
        player_rect = pygame.Rect(
            self.player_x, self.player_y, 
            self.player_width, self.player_height
        )
        
        for obstacle in self.obstacles:
            if not obstacle["passed"]:
                obstacle_rect = pygame.Rect(
                    obstacle["x"], 
                    self.ground_height - obstacle["height"], 
                    self.obstacle_width,
                    obstacle["height"]
                )
                if player_rect.colliderect(obstacle_rect):
                    return True
        
        return False
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Render the current game state"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("T-Rex Runner")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((255, 255, 255))
        
        # Draw ground
        pygame.draw.line(
            canvas, (0, 0, 0), 
            (0, self.ground_height), 
            (self.width, self.ground_height), 
            2
        )
        
        # Draw player
        pygame.draw.rect(
            canvas, (50, 50, 200),
            (self.player_x, self.player_y, self.player_width, self.player_height)
        )
        
        # Draw visible obstacles
        for obstacle in self.obstacles:
            if 0 <= obstacle["x"] <= self.width:
                pygame.draw.rect(
                    canvas, (200, 50, 50),
                    (
                        obstacle["x"], 
                        self.ground_height - obstacle["height"], 
                        self.obstacle_width,
                        obstacle["height"]
                    )
                )
        
        # Draw goal if visible
        if 0 <= self.goal_x <= self.width:
            pygame.draw.rect(
                canvas, (50, 200, 50),
                (self.goal_x, self.ground_height - 100, 50, 100)
            )
        
        # Draw UI elements
        if self.render_mode == "human":
            font = pygame.font.Font(None, 36)
            text = font.render(f"Score: {self.score}", True, (0, 0, 0))
            canvas.blit(text, (10, 10))
            
            controls_font = pygame.font.Font(None, 24)
            # controls_text = controls_font.render("A: Jump", True, (0, 0, 0))
            canvas.blit(controls_text, (10, 50))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
