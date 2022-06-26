#!/usr/bin/env python3

# import mujoco_py    
import rospkg
import time
import numpy as np
# from gym import spaces
import mujoco_py
from bipedal_env_v4 import walkerEnv4
from stable_baselines3 import PPO
rospk = rospkg.RosPack()
pkg_path = rospk.get_path('final_walker_mujoco')
model_path = pkg_path + "/ppo/2m_nsteps_PPO_local.zip"
# env = gym.make('walker-v1')
env = walkerEnv4()

steps = 1000
n_updates = 200

start = time.time()

model = PPO.load(model_path, env=env)
env.viewer = mujoco_py.MjViewer(env.sim)
for i in range(steps):
    # viewer.render()
    # sim.step()
    obs = env.reset()
    done = False
    # print(f'Checking if the state is part of the observation space: {env.observation_space.contains(state)}')
    # print(np.shape(state))
    for i in range(n_updates):
        env.viewer.render()
        actions, _ = model.predict(obs, deterministic=True)
        # actions = env.action_space.sample()
        # print(actions)
        obs, rew, done, info = env.step(actions)
        # print(actions)
        # print(obs[-6:])
        if done:
            break
# print()
end = time.time()
# print(end - start)

# viewer.finish()
viewer = None