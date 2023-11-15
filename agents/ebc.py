import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from copy import deepcopy

class EBCAgent(BaseAgent):
    def __init__(self, actor, critic, expert_demos, *args, **kwargs):
        super(EBCAgent, self).__init__(*args, **kwargs)
        self.actor = None
        self.critic = None
        self.expert_demos = expert_demos  # A list of (state, action) tuples
        self.actor_optimizer = None
        self.critic_optimizer = None
        # assert demon_l in ['mean', 'pi']
        # self.demon_l = demon_l

    def initNetwork(self, actor, critic, initialize_target=True):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        if initialize_target:
            self.actor_target = deepcopy(actor).to(self.device)
            self.critic_target = deepcopy(critic).to(self.device)
            self.target_networks = [self.actor_target, self.critic_target]
        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)

    def getSaveState(self):
        # Save the current state of the networks and optimizers
        save_state = super().getSaveState()
        # Additional state information can be added here if needed
        return save_state

    def loadFromState(self, save_state):
        # Load the saved state into the networks and optimizers
        super().loadFromState(save_state)
        # Additional loading actions can be performed here if needed
    
    def targetHardUpdate(self):
        """
        Hard update the target networks
        """
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def targetSoftUpdate(self):
        # Perform soft update of target networks
        # Target networks are assumed to be used and initialized
        tau = self.tau
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def getEGreedyActions(self, state, obs, eps):
        # Here, you would implement your method to get epsilon-greedy actions
        # For EBC, you would typically use the expert actions with some probability
        raise NotImplementedError("This method needs to be implemented.")

    def getGreedyActions(self, state, obs):
        # Here, you would implement your method to get greedy actions based on the learned policy
        raise NotImplementedError("This method needs to be implemented.")

    def _loadLossCalcDict(self):
        # This method is used to prepare the necessary tensors for loss calculation
        # You would unpack your batch and load it onto the device here
        raise NotImplementedError("This method needs to be implemented.")

    def calcActorLoss(self):
        # Calculate the actor loss, possibly including the behavior cloning loss
        # This will involve calculating the difference between the predicted actions and the expert actions
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pi, log_pi, mean = self.actor.sample(obs)
        self.loss_calc_dict['pi'] = pi
        self.loss_calc_dict['mean'] = mean
        self.loss_calc_dict['log_pi'] = log_pi

        qf1_pi, qf2_pi = self.critic(obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        pi = self.loss_calc_dict['pi']
        mean = self.loss_calc_dict['mean']
        action = self.loss_calc_dict['action_idx']
        is_experts = self.loss_calc_dict['is_experts']
        # add expert loss
        if is_experts.sum():
            if self.demon_l == 'pi':
                demon_loss = F.mse_loss(pi[is_experts], action[is_experts])
                # policy_loss += self.expert_demos * demon_loss
                policy_loss = self.expert_demos * demon_loss
            else:
                demon_loss = F.mse_loss(mean[is_experts], action[is_experts])
                # policy_loss += self.expert_demos * demon_loss
                policy_loss = self.expert_demos * demon_loss
        return policy_loss

    def calcCriticLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_obs)
            next_state_log_pi = next_state_log_pi.reshape(batch_size)
            qf1_next_target, qf2_next_target = self.critic_target(next_obs, next_state_action)
            qf1_next_target = qf1_next_target.reshape(batch_size)
            qf2_next_target = qf2_next_target.reshape(batch_size)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + non_final_masks * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(obs, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = qf1.reshape(batch_size)
        qf2 = qf2.reshape(batch_size)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        with torch.no_grad():
            td_error = 0.5 * (torch.abs(qf2 - next_q_value) + torch.abs(qf1 - next_q_value))
        return qf1_loss, qf2_loss, td_error
    
    def updateActorAndAlpha(self):
        # Update the actor network and the temperature parameter alpha if you're using entropy regularization
        policy_loss = self.calcActorLoss()
        log_pi = self.loss_calc_dict['log_pi']

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss

    def updateCritic(self):
        # Update the critic network
        qf1_loss, qf2_loss, td_error = self.calcCriticLoss()
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        return qf1_loss, qf2_loss, td_error
    
    def update(self, batch):
        # Perform an update step using the provided batch of experience
        # This method ties together actor and critic updates, and applies the behavior cloning
        self._loadBatchToDevice(batch)
        # qf1_loss, qf2_loss, td_error = self.updateCritic()
        # policy_loss = self.updateActorAndAlpha()

        actor_loss = self.calcActorLoss(batch)
        critic_loss = self.calcCriticLoss(batch)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        expert_states, expert_actions = zip(*self.expert_demos)
        expert_states = torch.tensor(expert_states, dtype=torch.float32, device=self.device)
        expert_actions = torch.tensor(expert_actions, dtype=torch.float32, device=self.device)
        
        predicted_actions = self.actor(expert_states)
        
        # Calculate the behavior cloning loss
        bc_loss = F.mse_loss(predicted_actions, expert_actions)

        # Backpropagate and update the actor network with behavior cloning loss
        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()

        # Perform any required soft updates
        self.targetSoftUpdate()

        return actor_loss.item(), critic_loss.item(), bc_loss.item()