# import tensorflow as tf 
# import itertools  #迭代工具
# import numpy as np 
import os 
# import sys 
# import matplotlib.pyplot as plt


# class StateProcessor(object):
#     """
#     获取当前状态。图像处理过程
#     """
#     def __init__(self):
#         # 建立tensorflow的图像
#         with tf.variable_creator_scope():
#             pass
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# import tensorflow as tf
# print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()