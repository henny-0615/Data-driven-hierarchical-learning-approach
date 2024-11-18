import math
from abc import ABC
import numpy as np
import gym
from gym import spaces
import torch


class ServoControlEnv(gym.Env, ABC):
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(9,))

        self.points = np.array([320, 640, 240, 480, 0, 0], dtype=np.int32)
        self.current_point_ind = -1
        self.zoom_rate = 1  # 1,3,6,10
        self.achieved_points = np.zeros(self.points.shape)
        self.achieve_error = 2

        self.min_dis = -1
        self.current_pos_x = 0
        self.current_pos_y = 0
        self.state = None
        self.x_done = False
        self.is_3d_located = False
        self.before_pos_x = 0
        self.before_pos_y = 0
        self.count_positioning_x = 0
        self.current_mean_x = 0
        self.count_positioning_y = 0
        self.current_mean_y = 0
        self.count_move = 0
        self.current_mean_move = 0

        self.set_x = np.zeros(751)
        self.set_y = np.zeros(561)

    def get_normalized_score(self, score):
        return (score - 0) / (500 - 0)

    def update_minDistance(self):
        if self.x_done:
            dis = abs(self.current_pos_y - self.points[self.current_point_ind + 1])
        else:
            dis = abs(self.current_pos_x - self.points[self.current_point_ind + 1])
        return dis

    def compute_reward(self):
        dis = self.update_minDistance()
        reward = 0

        if dis <= self.achieve_error:
            reward += 100
            self.current_point_ind += 1
            self.achieved_points[self.current_point_ind] = 1

            self.set_x = np.zeros(751)
            self.set_y = np.zeros(561)

            if self.achieved_points[0] and self.achieved_points[1]:
                self.x_done = True
        else:
            if not self.x_done:
                if self.is_3d_located:
                    if self.before_pos_x == 0:
                        self.min_dis = 320
                    else:
                        self.min_dis = 640 - self.before_pos_x

                reward += 1 / dis + 0.02 * (self.min_dis - dis) - 0.01 * abs(self.current_pos_x - self.before_pos_x) \
                          - self.set_x[self.current_pos_x] * 0.001
            else:
                if self.is_3d_located:
                    if self.before_pos_y == 0:
                        self.min_dis = 240
                    else:
                        self.min_dis = 480 - self.before_pos_y

                reward += 1 / dis + 0.02 * (self.min_dis - dis) - 0.01 * abs(self.current_pos_y - self.before_pos_y) \
                          - self.set_y[self.current_pos_y] * 0.001

            reward = min(reward, 99)

        return reward

    def step(self, action):
        assert self.state is not None, "Call reset before using step method."
        reward = 0
        done = False
        self.is_3d_located = False
        self.state = np.array(self.state, dtype=np.float32)
        
        if not self.x_done:
            positioning_max_dis = 320 + 110
            next_target = 0 if self.current_point_ind == -1 else self.points[self.current_point_ind]

            if abs(self.current_pos_x - next_target) <= self.achieve_error:
                self.is_3d_located = True

            self.current_pos_x = round(self.state[0] * 640 + action * positioning_max_dis)

            self.set_x[self.current_pos_x] += 1
        else:
            positioning_max_dis = 240 + 25
            next_target = 0 if self.current_point_ind == 1 else self.points[self.current_point_ind]

            if abs(self.current_pos_y - next_target) <= self.achieve_error:
                self.is_3d_located = True

            self.current_pos_y = round(self.state[1] * 480 + action * positioning_max_dis)

            self.set_y[self.current_pos_y] += 1

        reward += self.compute_reward()

        achieved_points = 0
        for i in range(6):
            if self.achieved_points[i]:
                achieved_points += 1
        if achieved_points >= 4:
            done = True
        else:
            self.before_pos_x = self.current_pos_x
            self.before_pos_y = self.current_pos_y

            if self.min_dis > self.update_minDistance():
                self.min_dis = self.update_minDistance()

        self.state = np.concatenate([np.array([self.current_pos_x / 640]), np.array([self.current_pos_y / 480]),
                                     self.achieved_points, np.array([self.zoom_rate / 10])])

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.current_pos_x = 0
        self.current_pos_y = 0
        self.before_pos_x = 0
        self.before_pos_y = 0
        self.x_done = False
        self.is_3d_located = False
        self.count_positioning_x = 0
        self.current_mean_x = 0
        self.count_positioning_y = 0
        self.current_mean_y = 0
        self.count_move = 0
        self.current_mean_move = 0
        self.min_dis = -1

        self.set_x = np.zeros(751)
        self.set_y = np.zeros(561)

        if np.random.uniform(0, 1) < 0.5:
            self.zoom_rate = 1
        else:
            self.zoom_rate = 6

        self.current_point_ind = -1
        self.achieved_points = np.zeros(self.points.shape)

        self.state = np.concatenate([np.array([self.current_pos_x / 640]), np.array([self.current_pos_x / 480]),
                                     self.achieved_points, np.array([self.zoom_rate / 10])])

        return np.array(self.state, dtype=np.float32)
