#!/usr/bin/env python3

import time
import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env
import gym
from gym import utils, spaces
# import rospkg
# from geometry_msgs.msg import Vector3
# import tf
np.set_printoptions(suppress=True)
import math

# rospk = rospkg.RosPack()
# pkg_path = rospk.get_path('biped_rl_mujoco')
# model_path = pkg_path + "/mujoco_model/new_model.xml"

model_path = '/content/drive/MyDrive/rl_walker/mujoco_model/new_model.xml'




class walkerEnv2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, exclude_current_positions_from_observation=True):
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        self.model_path = model_path
        # frame_skip = 10
        super(walkerEnv2).__init__()
        self.model = mujoco_py.load_model_from_path(model_path)
        self.width = 1000
        self.height = 1000
        self.sim = mujoco_py.MjSim(self.model)
        #self.viewer = mujoco_py.MjViewer(self.sim)
        # self.action_space = spaces.Box(high=0.4, low=-0.4, shape=(8,), dtype=np.float32)
        self.action_throttle = 1000
        self.action_space = spaces.MultiDiscrete([2000,2000, 2000, 2000, 2000, 2000, 2000, 2000])
        # self.action_space = spaces.Discrete(8)
        self.observation_space= spaces.Box(high=np.inf, low= -np.inf, shape=(40,))
        self.frame_skip = 10
        ############ class attributes ###########
        self.healthy_z_range = (0.2, 0.5)
        self.abs_max_roll = 1.6
        self.abs_max_pitch = 1.6
        self.x_velocity, self.y_velocity, self.z_velocity =  0, 0, 0 
        self.all_joint_vels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.list_of_joints = ['right_hip_x', 'right_hip_z', 'right_hip_y', 'knee_right', 'left_hip_x', 'left_hip_z', 'left_hip_y', 'knee_left']
        ########################################################3

        self.init_qpos = np.array(
            [0, 0, 0.4, 0.707107, 0, 0, 0.707107,\
            0, 0, 0, 0 ,0,\
            0, 0, 0, 0, 0], dtype=np.float32)
        

        self.init_qvel = np.array([
            0, 0, 0, 0, 0, 0,\
            0, 0, 0, 0, 0, 0,\
            0, 0, 0, 0 ], dtype=np.float32)

        
        ############## set all weights ###########
        self.terminate_when_unhealthy = True
        self.healthy_reward_weight = 2
        self.forward_reward_weight = 10
        self.ctrl_cost_weight = 0.2
        self.z_pos_reward = 50
        self.orientation_cost_weight = 1
        self.reset_noise_scale = 0.01

 
    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


    def mass_center(self, model, sim):
        mass = np.expand_dims(model.body_mass, 1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0:3].copy()

    def get_base_rpy(self):
        # euler_rpy = Vector3()
        x, y, z = self.euler_from_quaternion(self.sim.data.qpos[4], self.sim.data.qpos[5], self.sim.data.qpos[6], self.sim.data.qpos[3])
        # euler = tf.transformations.euler_from_quaternion([])
        euler_rpy_x = x
        euler_rpy_y = y
        euler_rpy_z = z
        return euler_rpy_x,euler_rpy_y,euler_rpy_z

    def walker_orientation_ok(self):
        x, y, z = self.get_base_rpy()
        roll_ok = self.abs_max_roll > abs(x)
        pitch_ok = self.abs_max_pitch > abs(y)
        orientation_ok = roll_ok and pitch_ok
        return orientation_ok
    
    def get_z_value_reward(self):
        return self.sim.data.qpos[2]*self.z_pos_reward 

    
    def get_observations(self):
        data = self.sim.data
        position = data.qpos.flat.copy()
        angular_position = [position[7:][self.sim.model.get_joint_qpos_addr(x)-7] for x in self.list_of_joints]
        angular_velocities = [self.all_joint_vels[self.sim.model.get_joint_qpos_addr(x)-7] for x in self.list_of_joints]

        x, y, z = self.get_base_rpy()
        return np.concatenate((
                                np.array([position[1], position[2],
                                self.x_velocity, self.y_velocity, self.z_velocity,
                                x, y, z]),
                                np.array(angular_position),
                                np.array(angular_velocities),
                                # place_holder
                                self.true_actions,
                                self.previous_actions,
        ))

    ################################################
    ########## Properties ##########################

    @property
    def healthy_reward(self):
        return (float(self.is_healthy or self.terminate_when_unhealthy)* self.healthy_reward_weight)

    @property
    def is_healthy(self):
        min_z, max_z = self.healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z
        return is_healthy

    @property
    def done(self):
        done_1 = (not self.is_healthy) if self.terminate_when_unhealthy else False
        return done_1

    ##################################################
    ############  Get reward functions ###############

    def actuator_control_cost(self):
        control_cost = self.ctrl_cost_weight * np.sum(abs(self.sim.data.qfrc_actuator.flat.copy()))
        return control_cost

    def get_reward(self):
        ############ Get obs reward ################
        x, y, z = self.get_base_rpy()
        orientation_cost = abs(x) + abs(y) + abs(z)
        ############################################

        r1 = self.actuator_control_cost()
        r2 = self.forward_reward_weight * self.x_velocity
        r3 = self.healthy_reward
        r4 = orientation_cost*self.orientation_cost_weight
        # r5 = self.get_z_value_reward()

        t_total = -r1 + r2 + r3 - r4
        return t_total

    def is_done(self):
        done_1 = self.done
        done_2 = not self.walker_orientation_ok()
        # print(done_1, done_2)
        # done_3 = False
        return (done_1 or done_2)
        
    ##################################################

    def step(self, action):
        # converting action space from 0 to 2000 to -1000 to 1000
        # print(action - 1000)
        action = action - 1000
        #++++++++++++++++++++++#
        # self.viewer.render()
        #++++++++++++++++++++++#
        self.previous_actions = action
        #=============================#
        # normalizing actions here
        self.true_actions = action*0.4/self.action_throttle
        # print(self.true_actions)
        ###############################################
        t1 = time.time()
        xyz_position_before = self.mass_center(self.model, self.sim)
        major_angles_before = self.sim.data.qpos[7:].copy()
        #--------------------------------------#
        self.do_simulation(self.true_actions, self.frame_skip)
        #--------------------------------------#
        xyz_position_after = self.mass_center(self.model, self.sim)
        major_angles_after = self.sim.data.qpos[7:].copy()
        t2 = time.time()
        ###############################################

        xyz_velocity = (xyz_position_after - xyz_position_before) / (t2 - t1)
        self.x_velocity, self.y_velocity, self.z_velocity = xyz_velocity
        self.all_joint_vels = (major_angles_after - major_angles_before)/(t2-t1)
        
        reward = self.get_reward()
        obs = self.get_observations()
        # print(obs)
        done = self.is_done()
        if done:
            return obs, -500, done, {}
        
        return obs, reward, done, {}

    def reset_model(self):    
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale
        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel  + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.previous_actions = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        self.true_actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.set_state(qpos,qvel)
        # print('Reset Called')
        return self.get_observations()
    
    # def reset(self):
    #     val = super().reset()
    #     obs = self.reset_model()
    #     print(obs)
    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 1
    #     self.viewer.cam.distance = self.model.stat.extent * 1.0
    #     self.viewer.cam.lookat[2] = 2.0
    #     self.viewer.cam.elevation = -20
