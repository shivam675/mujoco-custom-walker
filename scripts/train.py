#!/usr/bin/env python3


from bipedal_env_v3 import walkerEnv3
from stable_baselines3 import DDPG, TD3
# from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import gym
import rospkg
rospack = rospkg.RosPack()
# env = make_vec_env(walkerEnv3, n_envs=10)





if __name__ == "__main__":
    env = walkerEnv3()
    env = DummyVecEnv([lambda: env])
    # k = check_env(env, warn= True, skip_render_check=True)

    pkg_path = rospack.get_path('final_walker_mujoco')
    log_path =  pkg_path+'/training_results'
    # indir = pkg_path + '/training_results/m_nsteps_ppo.zip'
    outdir_2 = pkg_path + '/training_results/{}m_nsteps_ppo.zip'

    policy_kwargs = dict(net_arch=dict(pi=[512, 256, 256, 256], qf=[512, 256, 256, 256]))

    model = TD3('MlpPolicy', env, verbose=2,  learning_rate=0.001, tensorboard_log=log_path, policy_kwargs=policy_kwargs)
    # model = PPO.load(indir, env= env)
    for i in range(1, 20):
        model.learn(total_timesteps= 1000000, log_interval=100)
        model.save(outdir_2.format(i))