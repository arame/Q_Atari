import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, filename, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, filename)
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, stride=2)
        self.conv2 = nn.Conv2d(64, 64, stride=1)
        input_size = self.get_input_size(input_dims)
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.optimiser = optim.RMSprop(self.paramters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def get_input_size(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state)) 
        conv2 = F.relu(self.conv2(conv1)) 
        conv3 = F.relu(self.conv3(conv2)) 
        # conv3 shape is batch size X n_filters X H X W
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)
        return actions

    def save_checkpoint(self):
        print("** Saving Checkpoint file: " + self.chkpt_file)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("** Loading Checkpoint file: " + self.chkpt_file)
        self.load_state_dict(T.load(self.chkpt_file))

    
