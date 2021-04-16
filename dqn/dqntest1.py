import gym
from gym import Monitor
import itertools
import numpy as np 
import os 
import random
import sys 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_array_ops import gather
import datetime

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
        
    def _build_model(self):
        """
        建立网络模型
        """
        # 建立输入图像的变量
        self.x_pl=tf.Variable(tf.zeros(shape=[None,84,84,4]),name='X')
        # target value
        self.y_pl=tf.Variable(tf.zeros(shape=[None]),name='y')
        # 选择动作的序号
        self.actions_pl=tf.Variable(tf.zeros(shape=None),name='actions')
        # 映射像素值大小
        X=tf.cast(self.x_pl,tf.float32)/255.0 
        bat_size=tf.shape(self.x_pl)[0]
        model=tf.keras.Model([
            # 卷积层
            tf.keras.layers.Conv2D(32,8,4,activation='relu'),
            tf.keras.layers.Conv2D(64,4,2,activation='relu'),
            tf.keras.layers.Conv2D(64,3,1,activation='relu'),
            # 全连接层
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(len(VALID_ACTIONS))
        ])
        self.predictions=model(X)#执行模型的输出
        # 进行检验输出
        print(self.predictions)

        # 求动作函数的预测值,先求数据应该截取的位置
        incidents=tf.range(bat_size)*tf.shape(self.predictions)[1]+self.actions_pl

        # 先将predictions转换成一维向量然后切片
        self.act_predict=tf.gather(tf.reshape(self.predictions,[-1]),incidents)
        #训练参数
        model.compile(optimizer=tf.keras.optimizers.RMSprop(0.00025,0.99,0.0,1e-7),loss=tf.keras.losses.MSE(self.y_pl,self.act_predict))
        # 计算损失函数
        # self.losses=tf.math.squared_difference(self.y_pl,self.act_predict)
        # self.loss=tf.math.reduce_mean(self.losses)
        # 上面两步计算可以简化为直接一步计算
        # self.loss=tf.keras.losses.MSE(self.y_pl,self.act_predict)

        # RMSprop优化器
        # self.optimizer=tf.keras.optimizers.RMSprop(0.00025,0.99,0.0,1e-7)
        # self.train_op=self.optimizer.minimize(self.loss)

        # tensorboard显出训练总结
        log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            tf.summary.scalar("my_metric",0.5)
            writer.flush()
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # def predict(self,)
    # 这里需要考虑一下预测是怎么搞的