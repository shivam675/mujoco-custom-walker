#!/usr/bin/env python3

import time
import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env
import gym
from gym import utils, spaces
import rospkg
from geometry_msgs.msg import Vector3
import tf
np.set_printoptions(suppress=True)

rospk = rospkg.RosPack()
pkg_path = rospk.get_path('final_walker_mujoco')
model_path = pkg_path + "/model/fw_model.xml"





class walkerEnv3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, exclude_current_positions_from_observation=True):
        self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)
        self.model_path = model_path
        # frame_skip = 10
        super(walkerEnv3).__init__()
        self.model = mujoco_py.load_model_from_path(self.model_path)
        self.width = 1000
        self.height = 1000
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.action_space = spaces.Box(high=1, low=-1, shape=(6,), dtype=np.float32)
        self.action_throttle = 1000
        # self.action_space = spaces.MultiDiscrete([2000,2000, 2000, 2000, 2000, 2000])
        # self.action_space = spaces.Discrete(8)
        self.observation_space= spaces.Box(high=np.inf, low= -np.inf, shape=(29,))
        self.frame_skip = 5
        self.step_number = 0 
        ############ class attributes ###########
        self.healthy_z_range = (0.14, 0.45)
        self.abs_max_roll = 1.6
        self.abs_max_pitch = 1.6
        self.x_velocity, self.y_velocity, self.z_velocity =  0, 0, 0 
        self.all_joint_vels = [0, 0, 0, 0, 0, 0,]
        self.body_rpy_velocities = np.array([0, 0, 0])
        self.list_of_joints = ['right_hip', 'right_knee', 'left_hip', 'left_knee', 'left_ankle', 'right_ankle']

        ########################################################3

        self.init_qpos = np.array([ 0, 0, 0.34, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.init_qvel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        
        ############## set all weights ###########
        self.r6 = 0
        self.terminate_when_unhealthy = True
        self.healthy_reward_weight = 2
        self.forward_reward_weight = 10
        self.ctrl_cost_weight = 0.2
        self.z_pos_reward = 50
        self.orientation_cost_weight = 1
        self.reset_noise_scale = 0.01


    def mass_center(self, model, sim):
        mass = np.expand_dims(model.body_mass, 1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0:3].copy()

    def get_base_rpy(self):
        euler_rpy = Vector3()
        euler = tf.transformations.euler_from_quaternion([self.sim.data.qpos[4], self.sim.data.qpos[5], self.sim.data.qpos[6], self.sim.data.qpos[3]])
        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]
        return euler_rpy

    def walker_orientation_ok(self):
        orientation_rpy = self.get_base_rpy()
        roll_ok = self.abs_max_roll > abs(orientation_rpy.x)
        pitch_ok = self.abs_max_pitch > abs(orientation_rpy.y)
        orientation_ok = roll_ok and pitch_ok
        return orientation_ok
    
    def get_z_value_reward(self):
        return self.sim.data.qpos[2]*self.z_pos_reward 

    
    def get_observations(self):
        data = self.sim.data
        position = data.qpos.flat.copy()
        # position = data.qfrc_actuator.flat.copy()
        angular_position = [position[7:][self.sim.model.get_joint_qpos_addr(x)-7] for x in self.list_of_joints]
        angular_velocities = [self.all_joint_vels[self.sim.model.get_joint_qpos_addr(x)-7] for x in self.list_of_joints]

        angular_orientation = self.get_base_rpy()
        return np.concatenate((
                                np.array([position[1], position[2],
                                self.x_velocity, self.y_velocity, self.z_velocity,
                                angular_orientation.x, angular_orientation.y, angular_orientation.z,
                                self.body_rpy_velocities[0], self.body_rpy_velocities[1], self.body_rpy_velocities[2]]),
                                np.array(angular_position),
                                np.array(angular_velocities),
                                # place_holder
                                # self.true_actions,
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
        orientation_rpy = self.get_base_rpy()
        orientation_cost = abs(orientation_rpy.x) + abs(orientation_rpy.y) + abs(orientation_rpy.z)
        ############################################
        if (self.step_number >= 100) and (self.x_velocity <=2.0):
            self.r6 += self.step_number
            self.r6 /= 10

        r1 = self.actuator_control_cost()
        r2 = self.forward_reward_weight * self.x_velocity
        r3 = self.healthy_reward
        r4 = orientation_cost*self.orientation_cost_weight
        r5 = self.get_z_value_reward()

        t_total = -r1 + r2 + r3 - r4 -r5 - self.r6
        # print(t_total)
        return t_total

    def is_done(self):
        done_1 = self.done
        done_2 = not self.walker_orientation_ok()
        done_3 = False
        # print(self.step_number, self.x_velocity)
        if (self.step_number  > 1000) and (self.x_velocity < 2.0):
            done_3 = True
        
        # if self.step_number >= 2000 and self.x_velocity <= 
        # print(done_1, done_2, done_3)
        
        # done_3 = False
        return (done_1 or done_2 or done_3)
        
    ##################################################

    def step(self, action):
        #++++++++++++++++++++++#
        self.viewer.render()
        self.step_number += 1
        #++++++++++++++++++++++#
        self.previous_actions = action
        #=============================#
        # normalizing actions here
        # self.true_actions = action*0.4/self.action_throttle
        # print(self.true_actions)
        ###############################################
        t1 = time.time()
        xyz_position_before = self.mass_center(self.model, self.sim)
        major_angles_before = self.sim.data.qpos[7:].copy()
        rpy_before = self.get_base_rpy()
        #--------------------------------------#
        self.do_simulation(action, self.frame_skip)
        #--------------------------------------#
        xyz_position_after = self.mass_center(self.model, self.sim)
        major_angles_after = self.sim.data.qpos[7:].copy()
        rpy_after = self.get_base_rpy()
        t2 = time.time()
        ###############################################
        t_diff = t2-t1
        xyz_velocity = (xyz_position_after - xyz_position_before) / (t_diff)
        self.x_velocity, self.y_velocity, self.z_velocity = xyz_velocity
        self.all_joint_vels = (major_angles_after - major_angles_before)/(t_diff)
        self.body_rpy_velocities = np.array([rpy_after.x - rpy_before.x, rpy_after.y - rpy_before.y, rpy_after.z - rpy_before.z])/(t_diff)

        reward = self.get_reward()
        obs = self.get_observations()
        # print(obs)
        done = self.is_done()
        if done:
            return obs, -50000, done, {}
        
        return obs, reward, done, {}

    def reset_model(self):    
        self.step_number = 0
        self.r6 = 0
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale
        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel  + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.previous_actions = np.array([0, 0, 0, 0, 0, 0,], dtype=np.float32)
        # self.true_actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
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