import math
import sys
import time

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class TRexEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, num_obstacles=10):
        self.width = 800  # Screen width
        self.height = 400  # Screen height
        self.ground_height = 300  # Height of the ground from top
        
        # Jump heights (from ground)
        self.jump_height = 100
        self.squat_time = 20

        # Player properties
        self.player_width = 40
        self.player_height = 60
        self.player_x = 100  # Fixed x position
        
        # Obstacle properties
        self.num_obstacles = num_obstacles
        self.obstacle_width_jump = 30
        self.obstacle_height_jump = 50
        self.obstacle_width_squat = 30
        self.obstacle_height_squat = 120
        self.obstacle_squat_ground = 40
        
        # Obstacle spacing - carefully calculated based on jump physics
        self.min_gap = 250  # Minimum gap between obstacles
        self.max_gap = 450  # Maximum gap between obstacles
        
        # Game speed (pixels per frame)
        self.speed = 10.0
        
        # Physics
        self.gravity = 1.0
        
        # Action space: 0=no jump|squat, 1=jump, 2 = squat
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [
        #   player_y_normalized,
        #   is_jumping (0, 1 or 2),
        #   next_obstacle_distance_normalized,
        #   next_obstacle_type (0, 1),
        #   second_obstacle_distance_normalized,
        #   second_obstacle_type (0, 1)
        # ]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 2, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Pygame setup
        self.window = None
        self.clock = None
        self.isopen = True
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset player
        self.player_height = 60
        self.player_y = self.ground_height - self.player_height
        self.player_vel_y = 0
        self.player_squat_time = 0
        self.is_jump_squat = 0
        # Generate obstacles
        self.obstacles = []
        self.generate_obstacles()
        
        # Game state
        self.done = False
        self.truncated = False
        self.score = 0
        self.distance_traveled = 0
        
        # First render
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}
    
    def calculate_min_gap_for_height(self):
        """Calculate minimum safe gap based on physics and jump heights"""
        # For higher obstacles, we need more reaction distance
        # Time to complete a jump = 2 * sqrt(2 * height / gravity)
        # Distance covered = speed * time
        jump_time = 2 * math.sqrt(2 * self.jump_height / self.gravity)
        min_distance = self.speed * jump_time * 1.2  # 20% safety margin
        
        return max(self.min_gap, int(min_distance))
    
    def generate_obstacles(self):
        # Start with the first obstacle just outside the screen
        x = self.width + 100  # Give player time to react
        
        # Track recent height patterns to ensure variety
        obstacle_types = []
        
        for i in range(self.num_obstacles):
            # Choose a height that creates an interesting pattern
            if len(obstacle_types) >= 2 and obstacle_types[-1] == obstacle_types[-2]:
                obs_type = 1 - obstacle_types[-1]
            else:
                # Weighted random selection - make higher obstacles slightly less common
                #weights = [0.4, 0.35, 0.25]  # 40% low, 35% medium, 25% high
                obs_type = self.np_random.choice(2)
            
            obstacle_types.append(obs_type)
            
            # Calculate minimum gap based on the previous obstacle's height
            # For the first obstacle or after low obstacles, the standard min_gap works
            # For higher obstacles, we need more space for the player to react and jump
            if i > 0:
                min_gap = 150
                #prev_height_idx = recent_heights[-2]
                #min_gap = self.calculate_min_gap_for_height(prev_height_idx)
            else:
                min_gap = self.min_gap
                
            # Vary the gap based on a uniform distribution
            gap = self.np_random.integers(min_gap, self.max_gap)
            
            # Adjust for difficulty progression - increase min and max gaps slightly as game progresses
            difficulty_factor = min(1.0 + (i * 0.05), 1.3)  # Up to 30% increase
            gap = int(gap * difficulty_factor)
            
            x += gap
            
            self.obstacles.append({
                "x": x,
                "type": obs_type,
                "passed": False
            })
            
        # Set goal after the last obstacle with enough space for a nice finish
        self.goal_x = self.obstacles[-1]["x"] + 500
    
    def _get_obs(self):
        # Player y position normalized (0 = ground, 1 = max jump height)
        max_jump_height = self.jump_height
        player_height_from_ground = self.ground_height - self.player_y - self.player_height
        player_y_norm = min(1.0, max(0.0, player_height_from_ground / max_jump_height))
        
        # Find the next 2 obstacles that haven't been passed
        next_obstacles = [obs for obs in self.obstacles if not obs["passed"]][:2]
        
        # Pad with dummy obstacles if needed
        while len(next_obstacles) < 2:
            next_obstacles.append({
                "x": self.width*2,  # Far away
                "type": 0,    # Low height (doesn't matter since it's far)
                "passed": False
            })
        
        obs = [player_y_norm, self.is_jump_squat]
        
        for obstacle in next_obstacles:
            # Calculate true distance from player's right edge to obstacle's left edge
            true_distance = obstacle["x"] - (self.player_x + self.player_width)
            
            # Normalize distance (max distance we care about is the width of the screen)
            # Values beyond 1.0 are capped to 1.0
            distance_norm = min(1.0, max(0.0, true_distance / self.width))
            
            obs.extend([distance_norm, obstacle["type"]])
            
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        # Move obstacles and update game state
        self.distance_traveled += self.speed
        
        # Process action (jump)
        # Action 0 is "no jump", actions 1 is jump, action 2 is squat
        if action == 1 and not self.is_jump_squat:  # Can only jump if not already jumping
            self.is_jump_squat = 1
            target_height = self.jump_height
            # Calculate velocity needed to reach the target height
            self.player_vel_y = -np.sqrt(2 * self.gravity * target_height)
            
        if action == 2 and not self.is_jump_squat:
            self.is_jump_squat = 2
            self.player_squat_time = self.squat_time
            self.player_height /= 2
            self.player_y = self.ground_height - self.player_height
            
        # Update player position
        if self.is_jump_squat == 1:
            self.player_y += self.player_vel_y
            self.player_vel_y += self.gravity
        
            # Check if landed
            if self.player_y >= self.ground_height - self.player_height:
                self.player_y = self.ground_height - self.player_height
                self.player_vel_y = 0
                self.is_jump_squat = 0
                
        if self.is_jump_squat == 2:
            self.player_squat_time -=1
            
            if self.player_squat_time == 0:
                self.is_jump_squat = 0
                self.player_height *= 2
                self.player_y = self.ground_height - self.player_height
        # Update obstacles
        reward = 1 - (action > 0) * 2

        for obstacle in self.obstacles:
            obstacle["x"] -= self.speed
            
            # Check if passed
            if not obstacle["passed"] and obstacle["x"] + self.obstacle_width_jump < self.player_x:
                obstacle["passed"] = True
                self.score += 1
                # Reward based on obstacle height - higher obstacles give more reward
                obstacle_reward = 50 
                reward += obstacle_reward
        
        # Check for collisions
        if self._check_collision():
            self.done = True
            reward -= 50  # Big penalty for collision
        
        # Check if reached the goal
        self.goal_x -= self.speed
        if self.goal_x <= self.player_x:
            self.done = True
            reward += 100  # Very big reward for reaching the goal
        
        
        # Truncate if all obstacles are passed or after too many steps
        # if all(obs["passed"] for obs in self.obstacles) or self.distance_traveled > 10000:
        #     self.truncated = True
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, self.done, self.truncated, {"score": self.score}
    
    def _check_collision(self):
        player_rect = pygame.Rect(
            self.player_x, self.player_y, 
            self.player_width, self.player_height
        )
        
        for obstacle in self.obstacles:
            
            if not obstacle["passed"]:
                if obstacle["type"] == 0:
                    obstacle_rect = pygame.Rect(
                        obstacle["x"], 
                        self.ground_height - self.obstacle_height_jump, 
                        self.obstacle_width_jump,
                        self.obstacle_height_jump
                    )
                    
                else:
                    obstacle_rect = pygame.Rect(
                            obstacle["x"], 
                            self.ground_height - self.obstacle_height_squat - self.obstacle_squat_ground, 
                            self.obstacle_width_squat,
                            self.obstacle_height_squat
                    )
                if player_rect.colliderect(obstacle_rect):
                    return True
        
        return False
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
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
        
        # Draw obstacles
        for obstacle in self.obstacles:
            if 0 <= obstacle["x"] <= self.width:  # Only draw visible obstacles
                if obstacle["type"] == 0:
                    pygame.draw.rect(
                        canvas, (200, 50, 50),
                        (
                            obstacle["x"], 
                            self.ground_height - self.obstacle_height_jump, 
                            self.obstacle_width_jump,
                            self.obstacle_height_jump
                        )
                    )
                    
                else:
                     pygame.draw.rect(
                        canvas, (200, 50, 50),
                        (
                            obstacle["x"], 
                            self.ground_height - self.obstacle_height_squat - self.obstacle_squat_ground, 
                            self.obstacle_width_squat,
                            self.obstacle_height_squat
                        )
                    )
                    
                
        
        # Draw goal
        if 0 <= self.goal_x <= self.width:
            pygame.draw.rect(
                canvas, (50, 200, 50),
                (self.goal_x, self.ground_height - 100, 50, 100)
            )
        
        # Draw score and controls
        if self.render_mode == "human":
            font = pygame.font.Font(None, 36)
            text = font.render(f"Score: {self.score}", True, (0, 0, 0))
            canvas.blit(text, (10, 10))
            
            # Draw controls guide
            controls_font = pygame.font.Font(None, 24)
            controls_text = controls_font.render("A: Jump", True, (0, 0, 0))
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
