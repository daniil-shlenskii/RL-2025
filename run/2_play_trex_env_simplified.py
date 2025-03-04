import pygame
import sys
import time

from src.envs.custom_envs.trex_env_simplified import TRexEnvSimplified

def main():
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("T-Rex Runner - Human Playtest")
    
    # Create environment
    env = TRexEnvSimplified(render_mode="human")
    observation, _ = env.reset()
    
    done = False
    truncated = False
    
    # Game loop
    clock = pygame.time.Clock()
    
    print("=== T-Rex Runner Human Playtest ===")
    print("Controls: A=Jump")
    print("Press ESC to exit")
    
    last_jump_time = 0
    jump_cooldown = 0.1  # seconds
    
    info = {"score": 0}  # Initialize info dictionary with default score
    
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
        
        # Check for jump key (with cooldown to prevent multiple jumps)
        if current_time - last_jump_time > jump_cooldown:
            if keys[pygame.K_a]:
                action = 1  # Jump
                last_jump_time = current_time
        
        # Take action
        observation, reward, done, truncated, info = env.step(action)
        
        # Display score
        pygame.display.set_caption(f"T-Rex Runner - Human Playtest | Score: {info['score']}")
        
        # Cap the frame rate
        clock.tick(env.metadata["render_fps"])
    
    print(f"Game Over! Final Score: {info['score']}")
    time.sleep(2)  # Give player time to see final state
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()