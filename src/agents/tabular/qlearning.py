import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  
import numpy as np
import pygame 
import time
from gymnasium.wrappers import RecordVideo  
from src.agents.base import Agent


class QLearningAgent(Agent):
    def __init__(
        self,
        env: gym.Env,
        discount_factor: float = 0.99,
        learning_rate: float = 0.1,
        epsilon: float = 0.1, 
        episodes: int = 1000,
        eval_episodes: int = 100, 
        patience: int = 500, 
        seed: int = 42
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  
        self.seed = seed
        self.episodes = episodes
        self.eval_episodes = eval_episodes  
        self.patience = patience  

        # Initialize Q-table
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

        # Store rewards for visualization
        self.train_rewards_log = []
        self.eval_rewards_log = []

        # Save directory
        self.save_dir = "Matrics_plots"
        os.makedirs(self.save_dir, exist_ok=True)

    def sample_actions(self, states: np.ndarray) -> np.ndarray:
        """ Choose actions using fixed epsilon-greedy policy for exploration. """
        actions = []
        for state in states:
            if np.random.rand() < self.epsilon:  
                action = self.env.action_space.sample()
            else: 
                action = np.argmax(self.Q[state, :])
            actions.append(action)
        return np.array(actions)

    def eval_actions(self, states: np.ndarray) -> np.ndarray:
        """ Choose actions by exploiting the learned Q-table (greedy policy). """
        return np.array([np.argmax(self.Q[state, :]) for state in states])

    def train(self):
        """ Train the agent dynamically with early stopping (but fixed exploration). """
        best_reward = float("-inf")
        patience_counter = 0 

        for episode in range(self.episodes):
            state, _ = self.env.reset(seed=self.seed + episode)
            done, truncated = False, False
            total_reward = 0

            while not (done or truncated):
                action = self.sample_actions([state])[0]
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Q-learning update rule
                self.Q[state, action] += self.learning_rate * (
                    reward + self.discount_factor * np.max(self.Q[next_state, :]) - self.Q[state, action]
                )

                state = next_state
                total_reward += reward

            # Store total reward
            self.train_rewards_log.append(total_reward)

            # Early stopping check
            if total_reward > best_reward:
                best_reward = total_reward
                patience_counter = 0  
            else:
                patience_counter += 1  

            # Stop training if no improvement for `self.patience` episodes
            if patience_counter >= self.patience:
                print(f"Early stopping at episode {episode} (Best Reward: {best_reward:.2f})")
                break  # Stop training

            # Print progress every 100 episodes
            if episode % 100 == 0:
                print(f"Episode {episode} | Total reward: {total_reward:.2f} | Epsilon (Fixed): {self.epsilon:.2f}")

        # Save training results
        self.plot_results(self.train_rewards_log, "Training", "q_learning_training.png")

        # Evaluate the trained agent
        self.evaluate()

        # Record and save gameplay as MP4
        self.record_video("agent_performance.mp4", eval_episodes=5)

    def evaluate(self, num_episodes: int = 100):
        """ Evaluate the trained agent for a specified number of episodes. """
        print("\nEvaluating the trained agent...")
        self.eval_rewards_log = []  

        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed + 10000 + episode)
            done, truncated = False, False
            total_reward = 0

            while not (done or truncated):
                action = self.eval_actions([state])[0]  
                next_state, reward, done, truncated, _ = self.env.step(action)
                state = next_state
                total_reward += reward

            # Store evaluation reward
            self.eval_rewards_log.append(total_reward)

        # Compute and print average evaluation reward
        avg_eval_reward = np.mean(self.eval_rewards_log)
        print(f"\nAverage Evaluation Reward: {avg_eval_reward:.2f}")

        # Plot and save evaluation results
        self.plot_results(self.eval_rewards_log, "Evaluation", "q_learning_evaluation.png")

        return avg_eval_reward

    def record_video(self, video_filename="agent_performance.mp4", eval_episodes=5, fps=10):
        """ Manually record and save an MP4 video of the trained agent using Pygame with slower playback. """
        print("\n🎥 Recording the agent's performance...")

        video_path = os.path.join(self.save_dir, video_filename)

        # Ensure the environment is properly reset
        state, _ = self.env.reset(seed=self.seed + 20000)

        # Get screen dimensions for video resolution
        screen_width, screen_height = self.env.screen.get_size()

        # Define the video writer with lower FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        video = cv2.VideoWriter(video_path, fourcc, fps, (screen_width, screen_height))

        for episode in range(eval_episodes):
            done, truncated = False, False
            while not (done or truncated):
                # Render the Pygame environment
                self.env.render()
                pygame.display.flip()  # Ensure the display updates

                # Capture the frame from Pygame
                frame = pygame.surfarray.array3d(self.env.screen) 
                frame = np.rot90(frame)  
                frame = np.flipud(frame) 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

                # Write frame to video
                video.write(frame)

                # Agent takes an action
                action = self.eval_actions([state])[0]  
                state, _, done, truncated, _ = self.env.step(action)

                time.sleep(0.1) 

            # Reset for the next episode
            state, _ = self.env.reset()

        # Release the video file
        video.release()
        print(f"✅ Agent's gameplay video saved at: {video_path}")

    def plot_results(self, rewards_log, mode, filename):
        """ Plot total rewards over episodes and save the plot. """
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_log, label=f"{mode} Reward per Episode", alpha=0.5)

        # Compute moving average for smoothing
        window_size = 100
        if len(rewards_log) >= window_size:
            moving_avg = np.convolve(rewards_log, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size - 1, len(rewards_log)), moving_avg, label=f"Moving Avg ({window_size} episodes)", color="red")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{mode} Progress of Q-Learning Agent")
        plt.legend()
        plt.grid()

        # Save plot
        plot_path = os.path.join(self.save_dir, filename)
        plt.savefig(plot_path)
        print(f"Plot saved at: {plot_path}")

        plt.show()
