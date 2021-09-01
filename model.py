import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, outdim=64):
        """Initialize parameters and build model.ssssssssssssssssssssssssssss
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        # self.layer = nn.Sequential(nn.Linear(state_size, outdim), nn.ReLU(inplace=True), nn.Linear(outdim, outdim), nn.ReLU(inplace=True), nn.Linear(outdim, action_size))

        self.fc1 = nn.Linear(state_size, outdim)
        self.fc2 = nn.Linear(outdim, outdim)
        self.fc3 = nn.Linear(outdim, action_size)
    def forward(self, state):
        """Build a network that maps state -> action values."""

        # y = self.layer(state)    

        y = self.fc1(state)
        y = F.leaky_relu(y)
        y = self.fc2(y)
        y = F.leaky_relu(y)
        y = self.fc3(y)
        return y
