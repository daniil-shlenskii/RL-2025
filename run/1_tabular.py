
from src.agents import QLearningAgent
from src.envs import Simple1DPathEnv

# Initialize environment and agent
env = Simple1DPathEnv(x_end=20)
agent = QLearningAgent(env, episodes=100_000)

# Train the agent
agent.train()

# Evaluate the agent
agent.env = Simple1DPathEnv(x_end=20, render_mode="human") 
print(agent.evaluate(10))
