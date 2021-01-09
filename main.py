import numpy as np
from preprocess_frame import PreprocessFrame
from repeat_action_and_max_frame import RepeatActionAndMaxFrame
from stackframes import Stackframes 

def main():
    print("Start Atari check")
    environment_name = "Pong-v0"
    stack_size = 19
    new_shape = (84, 84, 1)
    make_env(environment_name, stack_size, stack_size)
    print("End Atari check")

def make_env(environment_name, new_shape, stack_size):
    env = gym.make(environment_name)
    rep = RepeatActionAndMaxFrame(env, 4)
    prep = PreprocessFrame(new_shape)
    stak = Stackframes(stack_size)
    return env


if __name__ == "__main__":
    main()
