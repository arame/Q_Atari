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
        self.conv2 = nn.Conv2d(64, 4, stride=2)
        self.conv2 = nn.Conv2d(64, 3, stride=1)
        input_size = get_input_size()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x 
