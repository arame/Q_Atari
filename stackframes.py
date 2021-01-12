import numpy as np
import gym
import collections

class Stackframes(gym.ObservationWrapper):
    def __init__(self,env, stackSize):
        super(Stackframes, self).__init__(env)
        self.stackSize = stackSize
        low = env.observation_space.low.repeat(stackSize, axis=0)
        high = env.observation_space.high.repeat(stackSize, axis=0)
        self.obs_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.frameStack = collections.deque(maxlen=stackSize)

    def reset(self):
        self.frameStack.clear()
        obs = self.env.reset()
        for _ in range(len(self.frameStack.maxlen)):
            self.frameStack.append(obs)
        arrObs = np.array(self.frameStack).reshape(self.observation_space.low.shape)
        return arrObs

    def observation(self, obs):
        self.frameStack.append(obs)
        arrObs = np.array(self.frameStack).reshape(self.observation_space.low.shape)
        return arrObs