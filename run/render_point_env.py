import numpy as np

from src.envs import PointEnv

env = PointEnv(render_mode="human")
env.reset()

for _ in range(10000):
    action = np.random.uniform(-1, 1, size=(1,))
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        env.reset()

env.close()
