import numpy as np
import torch as T
from replay_buffer import ReplayBuffer
from network import DeepQNetwork

class DeepQAgent:
    def __init__(self, lr, n_actions, input_dims, chkpt_dir, epsilon, gamma, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, replace=1000, algo=None, env_name=None):
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.chkpt_dir = chkpt_dir
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)
        filename = self.env_name+"_"+self.algo+'_q_eval'
        self.q_eval = DeepQNetwork(lr, n_actions, filename, input_dims, chkpt_dir)
        filename = self.env_name+"_"+self.algo+'_q_next'
        self.q_next = DeepQNetwork(lr, n_actions, filename, input_dims, chkpt_dir)

    def get_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        new_states = T.tensor(new_state).to(self.q_eval.device)
        return states, actions, rewards, new_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimiser.zero_grad()
        self.replace_target_network()
        states, actions, rewards, new_states, dones = self.sample_memory()
        indicies = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indicies, actions]
        q_next = self.q_next.forward(new_states).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma* q_next
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimiser.step()
        self.learn_step_counter +=1
        self.decrement_epsilon()


