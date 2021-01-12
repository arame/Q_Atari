import numpy as np
import gym
from preprocess_frame import PreprocessFrame
from repeat_action_and_max_frame import RepeatActionAndMaxFrame
from stackframes import Stackframes 

def main():
    print("Start Atari check")
    environment_name = "Pong-v0"
    stack_size = 19
    shape = (84, 84, 1)
    make_env(environment_name, shape, stack_size)
    print("End Atari check")

def make_env(environment_name, shape=(84,84,1), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(environment_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(env, shape)
    env = Stackframes(env, repeat)
    return env


if __name__ == "__main__":
    main()
