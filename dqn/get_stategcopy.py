import numpy as np
import airsim
import cv2 as cv

# 获取airsim飞行状态
class FlyingState:
    def __init__(self,x,y,z):
        self.dest=(x,y,z)
        self.score = 0
        self.client=airsim.MultirotorClient()
        self.linkToAirsim()
        self.client.takeoffAsync().join()
        # self.client.moveToPositionAsync()自动移动到目的地
        # client take off


    def linkToAirsim(self):
        # 连接到airsim
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # connect to the AirSim simulator 
        print('Connected')

    def frame_step(self, input_actions):
        # 执行操作并获取帧
        reward = 0.1
        terminal = False
        client_state=self.client.getMultirotorState()
        client_pre_pos=client_state.kinematics_estimated.position

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # ['forward','back','roll_right','roll_left','yaw_left','yaw_right','higher','lower']
        if input_actions[0]==1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.2,0.0,0.7,1).join()
        elif input_actions[1] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,-0.2,0.0,0.7,1).join()
        elif input_actions[2] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.2,0.0,0.0,0.7,1).join()
        elif input_actions[3] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(-0.2,0.0,0.0,0.7,1).join()
        elif input_actions[4] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.0,15.0,0.8,1).join()
        elif input_actions[5] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.0,-15.0,0.8,1).join()
        elif input_actions[6] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.0,0.0,1.0,1).join()
        elif input_actions[7] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.0,0.0,0.5,1).join()

        # client state
        client_state=self.client.getMultirotorState()
        client_position=client_state.kinematics_estimated.position
        # position and crash
        Crash_info=client_state.collision.has_collided
        if Crash_info :
            terminal=True
            self.__init__()
            reward=-float('inf')
        # rewards:closer +1 or farther -1
        dis_pre=np.linalg.norm([self.dest[0]-client_pre_pos.x_val,self.dest[1]-client_pre_pos.y_val,self.dest[2]-client_pre_pos.z_val])
        dis_this=np.linalg.norm([self.dest[0]-client_position.x_val,self.dest[1]-client_position.y_val,self.dest[2]-client_position.z_val])
        if dis_this < dis_pre:
            reward=1
        else:
            reward=-1
        self.score+=reward
        # front camera scene
        image_data=self.client.simGetImage(0,airsim.ImageType.Scene)
        return image_data, reward, terminal, self.score

# responses=client.simGetImages([
#     airsim.ImageRequest(0,airsim.ImageType.Scene),
#     # 前视深度信息
#     airsim.ImageRequest(0,airsim.ImageType.DepthVis),
#     # bottom深度信息
#     airsim.ImageRequest(3,airsim.ImageType.DepthVis)
# ])

# # game_state = GameState()
# # for i in range(0,20000):
# #     a_t_to_game = np.zeros(2)
# #     action_index = random.randrange(2)
# #     a_t_to_game[action_index] = 1
# #     game_state.frame_step(a_t_to_game)