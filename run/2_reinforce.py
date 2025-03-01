import gymnasium as gym

from src.agents import ReinforceAgent
from src.networks.policy import DiscretePolicy
from src.utils import init_run

# run init
SAVE_PATH = "artifacts/2_reinforce.ckpt"
args = init_run()

# env init
env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1", render_mode="human")

# agent init
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = DiscretePolicy(obs_dim, action_dim, hidden_dims=[128])
agent = ReinforceAgent(env, policy_net)

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
