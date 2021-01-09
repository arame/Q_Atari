import numpy as np
import gym

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat):
        super(RepeatActionAndMaxFrame, self).__init__()
        self.env = env
        self.repeat = repeat
        self.frameBuffer = []

    def step(self):
        totalReward = 0
        done = False
        
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step() 
            totalReward += reward
            #np.insert(self.frameBuffer, 0, obs, axis = 0)
            self.frameBuffer.append(obs)
            if done:
                break
        
        arr = np.array(self.frameBuffer)
        maxFrame = np.maximum(arr)
        return maxFrame, totalReward, done, info

    def reset(self):
        obs, reward, done, info = self.env.step()
        self.frameBuffer = []
        self.frameBuffer.append(obs)
        return obs



    
