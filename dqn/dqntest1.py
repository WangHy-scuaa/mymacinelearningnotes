import gym
from gym import Monitor
import itertools
import numpy as np 
import os 
import random
import sys 
import tensorflow as tf

if "../" not in sys.path:
    sys.path.append("../")
# 添加上一级文件夹作为引入包的
# 这里应该不需要，但可以记住

from matplotlib import pyplot as plt
from collections import deque, namedtuple

# 倒立摆小车游戏
env=gym.make('CartPole-v0')
obv=env.reset() # 初始观察值
# 动作actions: 0:left,1:right
VALID_ACTIONS=[0,1]

# class StateProcessor(object):
#     """
#     处理图像元数据作为状态
#     """
#     def __init__(self) -> None:
#         super().__init__()
#         self.input_state=tf.constant()
class Estimator(object):
    """
    Q函数神经网络
    """
    def __init__(self) -> None:
        super().__init__()
        
