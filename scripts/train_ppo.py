#!/usr/bin/env python3

from bipedal_env_v4 import walkerEnv4
from stable_baselines3 import PPO
# from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import torch
import rospkg
rospack = rospkg.RosPack()
env = make_vec_env(walkerEnv4, n_envs=10)


if __name__ == "__main__":
    # env = walkerEnv3()
    # env = DummyVecEnv([lambda: env])
    # k = check_env(env, warn= True, skip_render_check=True)
    pkg_path = rospack.get_path('final_walker_mujoco')
    log_path =  pkg_path+'/ppo'
    # indir = pkg_path + '/ppo/4m_nsteps_PPO_local.zip'
    outdir_2 = pkg_path + '/ppo/{}m_nsteps_PPO_local.zip'
    # print(outdir_2.format(2))
    # policy_kwargs = dict(net_arch=dict(pi=[512, 256, 256, 256], qf=[512, 256, 256, 256]))
    policy_kwargs = dict(net_arch=[dict(pi=[512, 256, 256, 256], vf=[512, 256, 256, 256])])
    model = PPO(
        'MlpPolicy',
        env,
        verbose=2,  
        learning_rate=0.0003,
        tensorboard_log=log_path, 
        policy_kwargs=policy_kwargs, 
        n_steps=4096,
        # use_sde=True,
        # create_eval_env=True,
        seed=15)
    # model = PPO.load(indir, env= env,)
    
    for i in range(1, 100):
        # model.set_random_seed()
        model.learn(total_timesteps=2000000, 
        log_interval=1)
        model.save(outdir_2.format(i*2))
        # model.