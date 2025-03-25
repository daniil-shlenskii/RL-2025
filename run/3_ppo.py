import gymnasium as gym

from src.agents import PPOAgent
from src.networks.critic import StateCritic
from src.networks.policy import DiscretePolicy
from src.utils import init_run

# run init
SAVE_PATH = "artifacts/3_ppo"
env_name = "TRexJumpSquatEnv-v0"
args = init_run()

# env init
env = gym.make(env_name)
eval_env = gym.make(env_name, render_mode="human")

# agent init
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = DiscretePolicy(obs_dim, action_dim, hidden_dims=[128, 128])
value_net = StateCritic(obs_dim, hidden_dims=[128, 128])
agent = PPOAgent(
    env,
    policy_net=policy_net,
    value_net=value_net,
)

if not args.eval: # train
    save_every = 100 if args.save_ckpt else None
    agent.train(
        num_episodes=5_000,
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
