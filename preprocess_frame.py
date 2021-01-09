import numpy as np
import gym
import cv2

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, newShape):
        self.env = env
        self.newShape = newShape
    
        self.obs_space = gym.spaces.Box[i] for i in range(19)



    def observation(self, rawObs):
        greyObs = cv2.cvtColor(rawObs, cv2.COLOR_BGR2GRAY)
        newObs = np.arrange(greyObs).reshape(self.newShape)
        arrObs = np.array(newObs)
        arrObs = np.moveaxis(arrObs, -1, 0)
        arrObs /= 255
        return arrObs
