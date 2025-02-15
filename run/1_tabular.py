import gymnasium as gym

from src.agents import QLearningAgent

# Example usage
env = gym.make('CliffWalking-v0')
print(env)
agent = QLearningAgent(env)

# Train the agent
agent.train()

# Evaluate the agent
env = gym.make('CliffWalking-v0', render_mode='human')
agent.env = env
print(agent.evaluate(10))
