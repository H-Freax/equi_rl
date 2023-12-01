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
    def __init__(self, input_dim, output_dim, lr=1e-4, gamma=0.95, device='cuda', action_ranges=None, n_a=4, *args, **kwargs):
        super(BehaviorCloningAgent, self).__init__(lr, gamma, device, *args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_ranges = action_ranges  # Dictionary of action ranges, e.g., {'dx': (-1, 1), 'dy': (-1, 1)}
        self.n_a = n_a  # Number of action dimensions
        self.network = None

    def initNetwork(self, network, initialize_target=False):
        self.network = network.to(self.device)
        self.networks.append(self.network)
        self.bc_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.optimizers.append(self.bc_optimizer)
        if initialize_target:
            self.updateTarget()

    def update(self, batch):
        states, obs, actions, _, _, _, _, _, _ = self._loadBatchToDevice(batch)
        predicted_actions = self.network(obs)
        loss = F.mse_loss(predicted_actions, actions)
        self.bc_optimizer.zero_grad()
        loss.backward()
        self.bc_optimizer.step()
        return loss.item()

    def getEGreedyActions(self, state, obs, eps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).to(self.device).unsqueeze(0)
            predicted_action = self.network(obs_tensor)
            return predicted_action.squeeze(0).cpu().numpy()

    def getActionFromPlan(self, plan):
        def getUnscaledAction(action, action_range):
            unscaled_action = 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1
            return unscaled_action

        actions = {key: plan[:, idx].clamp(*self.action_ranges[key]) for idx, key in enumerate(self.action_ranges)}
        unscaled_actions = {key: getUnscaledAction(action, self.action_ranges[key]) for key, action in actions.items()}

        if self.n_a == 5:
            return self.decodeActions(*[unscaled_actions[key] for key in sorted(unscaled_actions.keys())])
        else:
            return self.decodeActions(*[unscaled_actions[key] for key in sorted(unscaled_actions.keys())][:-1])

    def decodeActions(self, *unscaled_actions):
        # Implement this method based on how you want to decode your actions
        # Example: return np.array(unscaled_actions)
        pass