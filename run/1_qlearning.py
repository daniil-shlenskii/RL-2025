import gymnasium as gym

from src.agents import QLearningAgent
from src.utils import init_run

# run init
SAVE_PATH = "artifacts/1_qlearning"
env_name = "TRex1dEnv-v0"
args = init_run()

# env init
env = gym.make(env_name)
eval_env = gym.make(env_name, render_mode="human")

# agent init
agent = QLearningAgent(env)

if not args.eval: # train
    save_every = 500 if args.save_ckpt else None
    agent.train(
        num_episodes=10_000,
        log_every=100,
        save_every=save_every,
        save_path=SAVE_PATH,
    )
else:
    agent.load(SAVE_PATH)

# evaluate
agent.env = eval_env
score = agent.evaluate(3)
print(f"eval score: {score}")
