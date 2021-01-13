import numpy as np
import gym
from preprocess_frame import PreprocessFrame
from repeat_action_and_max_frame import RepeatActionAndMaxFrame
from stackframes import Stackframes 
from utilz import plot_learning_curve
from agent import DeepQAgent

def main():
    print("Start Atari games")
    environment_name = "PongNoFrameskip-v4"
    env = make_env(environment_name)
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500
    lr = 0.0001
    epsilon = 1
    gamma = 0.99
    input_dims = (env.observation_space.shape)
    n_actions = env.action_space.n
    eps_min=0.01
    eps_dec=5e-7
    replace=1000, 
    algo=None
    mem_size = 50000
    batch_size = 32
    chkpt_dir = "models/"
    algo = "DeepQAgent"
    agent = DeepQAgent(lr, n_actions, input_dims, chkpt_dir, epsilon, gamma, mem_size, batch_size, eps_min, eps_dec, replace, algo, environment_name)
    if load_checkpoint:
        agent.load_models()
    fname = agent.algo + "_" + agent.env_name + '_lr' + str(agent.lr) + "_" + str(n_games) + "_games"
    figure_file = "plots/" +fname + ".png"
    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    for i in range(n_games):
        done = False
        score = 0
        observation =  env.reset()
        while not done:
            action = agent.get_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, new_observation, int(done))
                agent.learn()
            observation = new_observation
            n_steps +=1
        score.append(score)  
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])  
        print("episode ", i +  1, "score: ", score, "average score %.1f best score %.1f epsilon %.2f" %(avg_score, best_score, agent.epsilon), " steps ", n_steps)
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score
        eps_history.append(agent.epsilon)
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
    print("End Atari games")

def make_env(environment_name, shape=(84,84,1), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(environment_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(env, shape)
    env = Stackframes(env, repeat)
    return env


if __name__ == "__main__":
    main()
