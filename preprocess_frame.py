import numpy as np
import gym
import cv2

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2]. shape[0], shape[1])
        self.obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        newObs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        newObs /= 255
        return newObs
