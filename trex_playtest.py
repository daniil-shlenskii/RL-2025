import pygame
import sys
import os
import time

from src.envs.custom_envs.trex_env import TRexEnv

def main():
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("T-Rex Runner - Human Playtest")
    
    # Create environment
    env = TRexEnv(render_mode="human")
    observation, _ = env.reset()
    
    done = False
    truncated = False
    
    # Game loop
    clock = pygame.time.Clock()
    
    print("=== T-Rex Runner Human Playtest ===")
    print("Controls: A=Small Jump, S=Medium Jump, D=High Jump")
    print("Press ESC to exit")
    
    last_jump_time = 0
    jump_cooldown = 0.1  # seconds
    
    while not (done or truncated):
        current_time = time.time()
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
        
        # Process key inputs
        keys = pygame.key.get_pressed()
        
        # Default action is "no jump" (action 0)
        action = 0
        
        # Check for jump keys (with cooldown to prevent multiple jumps)
        if current_time - last_jump_time > jump_cooldown:
            if keys[pygame.K_a]:
                action = 1  # Small jump
                last_jump_time = current_time
            elif keys[pygame.K_s]:
                action = 2  # Medium jump
                last_jump_time = current_time
            elif keys[pygame.K_d]:
                action = 3  # High jump
                last_jump_time = current_time
        
        # Take action
        observation, reward, done, truncated, info = env.step(action)
        
        # Display score
        pygame.display.set_caption(f"T-Rex Runner - Human Playtest | Score: {env.score}")
        
        # Cap the frame rate
        clock.tick(30)
    
    print(f"Game Over! Final Score: {env.score}")
    time.sleep(2)  # Give player time to see final state
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()