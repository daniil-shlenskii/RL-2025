import gymnasium as gym

from src.agents import QLearningAgent
from src.utils import init_run

# run init
SAVE_PATH = "artifacts/1_tabular"
args = init_run()

# env init
env = gym.make("CliffWalking-v0")
eval_env = gym.make("CliffWalking-v0", render_mode="human")

# agent init
agent = QLearningAgent(env)

if not args.eval: # train
    save_every = 500 if args.save_ckpt else None
    agent.train(
        num_episodes=1500,
        log_every=100,
        save_every=save_every,
        save_path=SAVE_PATH,
    )

# evaluate
agent.load(SAVE_PATH)
agent.env = eval_env
score = agent.evaluate(1)
print(f"eval score: {score}")
