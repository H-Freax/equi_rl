from agents.base_agent import BaseAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BehaviorCloningNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(128, 256)):
        super(BehaviorCloningNetwork, self).__init__()
        self.fc1 = nn.Linear(np.prod(obs_dim), hidden_dims[0])  # Flattening the input
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BehaviorCloningAgent(BaseAgent):
    def __init__(self, input_dim, output_dim, lr=1e-4, gamma=0.95, device='cuda', *args, **kwargs):
        super(BehaviorCloningAgent, self).__init__(lr, gamma, device, *args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = None  # Placeholder for the network

    def initNetwork(self, network, initialize_target=False):
        self.network = network.to(self.device)
        self.networks.append(self.network)
        self.bc_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.optimizers.append(self.bc_optimizer)
        if initialize_target:
            self.updateTarget()

    def update(self, batch):
        states, obs, actions, _, _, _, _, _, _ = self._loadBatchToDevice(batch)

        # Forward pass through the network
        predicted_actions = self.network(obs)

        # Calculate loss (e.g., MSE for continuous actions)
        loss = F.mse_loss(predicted_actions, actions)

        # Backpropagation
        self.bc_optimizer.zero_grad()
        loss.backward()
        self.bc_optimizer.step()

        return loss.item()

    def getEGreedyActions(self, state, obs, eps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).to(self.device).unsqueeze(0)
            predicted_action = self.network(obs_tensor)
            return predicted_action.squeeze(0).cpu().numpy()

    # You might need to implement other methods as per your BaseAgent class
