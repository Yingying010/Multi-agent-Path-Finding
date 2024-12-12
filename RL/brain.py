import torch.nn as nn
import torch.nn.functional as F
import torch
import random


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layer):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for idx, net in enumerate(self.layers):
            if idx == len(self.layers) - 1:
                return net(x)
            else:
                x = F.relu(net(x))
        return x


class DQN(object):
    def __init__(self, input_size, hidden_size, output_size, epsilon_decay_factor, min_epsilon, gamma,
                 device, learning_rate, update_step=5):
        self.q_net = Net(input_size, hidden_size, output_size).to(device)
        self.target_q_net = Net(input_size, hidden_size, output_size).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = 1.0
        self.epsilon_decay_factor = epsilon_decay_factor
        self.min_epsilon = min_epsilon

        self.gamma = gamma
        self.device = device
        self.update_step = update_step
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.count = 0

    def update(self, transitions):
        states = transitions["states"].to(self.device)
        actions = torch.tensor(transitions["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = transitions["next_states"].to(self.device)
        done = torch.tensor(transitions["done"], dtype=torch.int64).view(-1, 1).to(self.device)

        q_value = self.q_net(states).gather(1, actions)
        next_q_value = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        target_q_value = next_q_value * self.gamma * (1 - done) + rewards

        loss = F.mse_loss(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % self.update_step == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_factor)

    def get_q_value(self, state):
        return self.target_q_net(state)
