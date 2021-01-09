import numpy as np
import gym

class Stackframes(gym.ObservationWrapper):
    def __init__(self,env, stackSize):
        self.env = env
        self.stackSize = stackSize
        self.obs_space = gym.spaces.Box[i] for i in range(19)
        self.frameStack = []

    def reset(self):
        self.frameStack = []
        obs, reward, done, info = self.env.reset()
        for i in range(len(self.frameStack)):
            self.frameStack[i] = obs
        arrObs = np.array(self.frameStack)
        arrObs = np.moveaxis(arrObs, -1, 0)
        return arrObs

    def observation(self, obs):
        self.frameStack.append(obs)
        arrObs = np.array(self.frameStack)
        arrObs = np.moveaxis(arrObs, -1, 0)
        return arrObs