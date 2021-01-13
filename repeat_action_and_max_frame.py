import numpy as np
import gym

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat, clip_rewards, no_ops, fire_first):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.shape = env.observation_space.low.shape
        self.repeat = repeat
        self.frameBuffer = np.zeros_like((2, self.shape))
        self.clip_rewards = clip_rewards
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        totalReward = 0.0
        done = False
        
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_rewards: 
                reward = np.clip(np.array([reward]), -1, 1)[0]
            totalReward += reward
            idx = i % 2
            self.frameBuffer[idx] = obs
            if done:
                break
        
        maxFrame = np.maximum(self.frameBuffer[0], self.frameBuffer[1])
        return maxFrame, totalReward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
            obs, _, _, _ = self.env.step(1)
        
        self.frameBuffer = np.zeros_like((2, self.shape))
        self.frameBuffer[0] = obs
        return obs



    
