import os
import sys
import torch

from src.agents import ReinforceAgent
from src.networks.policy import DiscretePolicy
from src.utils import init_run
from src.envs.custom_envs.trex_env import TRexEnv

# run init
SAVE_PATH = "artifacts/trex_reinforce.ckpt"
args = init_run()

# Create directories if they don't exist
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# env init
env = TRexEnv()
eval_env = TRexEnv(render_mode="human")

# agent init
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = DiscretePolicy(obs_dim, action_dim, hidden_dims=[128, 64])
agent = ReinforceAgent(env, policy_net)

if not args.eval:  # train
    save_every = 100 if args.save_ckpt else None
    agent.train(
        num_episodes=500,
        log_every=10,
        save_every=save_every,
        save_path=SAVE_PATH,
    )

# evaluate
agent.load(SAVE_PATH)
agent.env = eval_env
score = agent.evaluate(3)  # Evaluate on 3 episodes
print(f"eval score: {score}")