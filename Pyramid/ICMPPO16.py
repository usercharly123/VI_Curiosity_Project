import numpy as np
import torch
import torch.nn as nn
from ICM16 import ICM
from ActorCritic import ActorCritic
from utils import Swish, linear_decay_beta, linear_decay_lr, linear_decay_eps


class ICMPPO:
    def __init__(self, writer, state_dim=172, action_dim=5, n_latent_var=512, lr=3e-4, betas=(0.9, 0.999),
                 gamma=0.99, ppo_epochs=3, icm_epochs=1, eps_clip=0.2, ppo_batch_size=128,
                 icm_batch_size=16, intr_reward_strength=0.02, lamb=0.95, device='cpu', reward_mode='both'):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lambd = lamb
        self.eps_clip = eps_clip
        self.ppo_epochs = ppo_epochs
        self.icm_epochs = icm_epochs
        self.ppo_batch_size = ppo_batch_size
        self.icm_batch_size = icm_batch_size
        self.intr_reward_strength = intr_reward_strength
        self.device = device
        self.writer = writer
        self.timestep = 0
        self.reward_mode = reward_mode  # Can be 'both', 'extrinsic', or 'intrinsic'
        self.icm = ICM(activation=Swish()).to(self.device)

        self.policy = ActorCritic(state_dim=state_dim,
                                  action_dim=action_dim,
                                  n_latent_var=n_latent_var,
                                  activation=Swish(),
                                  device=self.device,
                                  ).to(self.device)
        self.policy_old = ActorCritic(state_dim,
                                      action_dim,
                                      n_latent_var,
                                      activation=Swish(),
                                      device=self.device
                                      ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=lr, betas=betas)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss(reduction='none')

    def update(self, memory, timestep, current_os='windows'):
        # Convert lists from memory to tensors
        self.timestep = timestep
        
        old_states = torch.tensor(
            np.array([t.detach().cpu().numpy() for t in memory.states]),
            dtype=torch.float32,
            device=self.device
        )
        old_actions = torch.tensor(
            np.array([t.detach().cpu().numpy() for t in memory.actions]),
            dtype=torch.float32,
            device=self.device
        )
        old_logprobs = torch.tensor(
            np.array([t.detach().cpu().numpy() for t in memory.logprobs]),
            dtype=torch.float32,
            device=self.device
        )

        old_states = torch.transpose(old_states, 0, 1)
        old_actions = torch.transpose(old_actions, 0, 1)
        old_logprobs = torch.transpose(old_logprobs, 0, 1)
            
        # Finding s, n_s, a, done, reward:
        curr_states = old_states[:, :-1, :] 
        next_states = old_states[:, 1:, :] 
        actions = old_actions[:, :-1].long()
        
        # Debug information
        print("Memory rewards length:", len(memory.rewards))
        print("First few rewards shapes:", [r.shape if hasattr(r, 'shape') else type(r) for r in memory.rewards[:5]])
        print("First few rewards:", [r for r in memory.rewards[:5]])
        
        # Process rewards based on OS
        if current_os == 'linux':
            # On Linux, rewards are 1D arrays, need to reshape
            rewards_list = [r.reshape(-1, 1) if r.ndim == 1 else r for r in memory.rewards[:-1]]
            rewards_np = np.array(rewards_list)  # Shape: (2047, 16, 1)
            rewards = torch.tensor(rewards_np).to(self.device).detach()  # Shape: (2047, 16, 1)
            rewards = rewards.permute(1, 0, 2)  # Shape: (16, 2047, 1)
        else:  # Windows
            # On Windows, rewards are already in correct shape
            rewards_np = np.array(memory.rewards[:-1])
            rewards = torch.tensor(rewards_np).to(self.device).detach()
        
        mask = (~torch.tensor(
            np.array(memory.is_terminals),
            dtype=torch.bool
        ).T[:, :-1].to(self.device)).long()
        
        with torch.no_grad():
            intr_reward, _, _ = self.icm(actions, curr_states, next_states, mask)
        intr_rewards = torch.clamp(self.intr_reward_strength * intr_reward, 0, 1)

        # Combine rewards based on reward_mode
        if self.reward_mode == 'both':
            # Ensure both rewards have the same shape
            if current_os == 'linux':
                # On Linux, rewards are (16, 2047, 1) and intr_rewards are (16, 2047)
                combined_rewards = (rewards.squeeze(-1) + 0.1*intr_rewards) / 2
            else:  # Windows
                # On Windows, both should already be in correct shape
                combined_rewards = (rewards + 0.1*intr_rewards) / 2
        elif self.reward_mode == 'extrinsic':
            combined_rewards = rewards
        elif self.reward_mode == 'intrinsic':
            combined_rewards = intr_rewards
        else:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}. Must be 'both', 'extrinsic', or 'intrinsic'")

        self.writer.add_scalar('Mean_intr_reward_per_1000_steps',
                               intr_rewards.mean() * 1000,
                               self.timestep
                               )

        # Finding cumulative advantage
        with torch.no_grad():
            state_values = torch.squeeze(self.policy.value_layer(curr_states))  # Shape: (16, 2047)
            next_state_values = torch.squeeze(self.policy.value_layer(next_states))  # Shape: (16, 2047)
            if current_os == 'linux':
                td_target = combined_rewards.squeeze(-1) + self.gamma * next_state_values * mask  # Shape: (16, 2047)
            else:
                td_target = combined_rewards + self.gamma * next_state_values * mask  # Shape: (16, 2047)
            delta = td_target - state_values  # Shape: (16, 2047)

            self.writer.add_scalar('maxValue',
                                   state_values.max(),
                                   timestep
                                   )
            self.writer.add_scalar('meanValue',
                                   state_values.mean(),
                                   self.timestep
                                   )

            advantage = torch.zeros(16, 1).to(self.device)      
            advantage_lst = []
            for i in range(delta.size(1) - 1, -1, -1):
                delta_t, mask_t = delta[:, i:i+1], mask[:, i:i+1]  # Keep dimensions
                advantage = delta_t + (self.gamma * self.lambd * advantage) * mask_t
                advantage_lst.insert(0, advantage)

            advantage_lst = torch.cat(advantage_lst, dim=1)  # Shape: (16, 2047)
            # Get local advantage to train value function
            local_advantages = state_values + advantage_lst
            # Normalizing the advantage
            advantages = (advantage_lst - advantage_lst.mean()) / (advantage_lst.std() + 1e-10)

        # Optimize policy for ppo epochs:
        epoch_surr_loss = 0
        for _ in range(self.ppo_epochs):
            indexes = np.random.permutation(actions.size(0))    # WAS actions.size(1)
            # Train PPO and icm
            for i in range(0, len(indexes), self.ppo_batch_size):
                batch_ind = indexes[i:i + self.ppo_batch_size]
                batch_curr_states = curr_states[:, batch_ind, :]    # WAS curr_states[:, batch_ind, :]
                batch_actions = actions[:, batch_ind]         # WAS actions[:, batch_ind]
                batch_mask = mask[:,batch_ind]       # WAS mask[:, batch_ind]
                batch_advantages = advantages[:, batch_ind]     # WAS advantages[:, batch_ind]
                batch_local_advantages = local_advantages[:, batch_ind]   # WAS local_advantages[:, batch_ind]
                batch_old_logprobs = old_logprobs[:, batch_ind]     # WAS old_logprobs[:, batch_ind]

                # Finding actions logprobs and states values
                batch_logprobs, batch_state_values, batch_dist_entropy = self.policy.evaluate(batch_curr_states,
                                                                                              batch_actions)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(batch_logprobs - batch_old_logprobs.detach())

                # Apply linear decay and multiply 16 times cause agents_batch is 16 long
                decay_epsilon = linear_decay_eps(self.timestep * 16)
                decay_beta = linear_decay_beta(self.timestep * 16)

                # Finding Surrogate Loss:
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - decay_epsilon, 1 + decay_epsilon) * batch_advantages
                loss = -torch.min(surr1, surr2) * batch_mask + \
                       0.5 * nn.MSELoss(reduction='none')(batch_state_values,
                                                           batch_local_advantages.detach()) * batch_mask - \
                       decay_beta * batch_dist_entropy * batch_mask
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                linear_decay_lr(self.optimizer, self.timestep * 16)

                epoch_surr_loss += loss.item()

        self._icm_update(self.icm_epochs, self.icm_batch_size, curr_states, next_states, actions, mask)
        self.writer.add_scalar('Lr',
                               self.optimizer.param_groups[0]['lr'],
                               self.timestep
        )
        self.writer.add_scalar('Surrogate_loss',
                               epoch_surr_loss / (self.ppo_epochs * (len(indexes) // self.ppo_batch_size + 1)),
                               self.timestep
        )

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _icm_update(self, epochs, batch_size, curr_states, next_states, actions, mask):
        epoch_forw_loss = 0
        epoch_inv_loss = 0
        for _ in range(epochs):
            indexes = np.random.permutation(actions.size(0))
            for i in range(0, len(indexes), batch_size):
                batch_ind = indexes[i:i + batch_size]
                batch_curr_states = curr_states[:, batch_ind, :]    # WAS curr_states[:, batch_ind, :]
                batch_next_states = next_states[:, batch_ind, :]    # WAS next_states[:, batch_ind, :]
                batch_actions = actions[:, batch_ind]      # WAS actions[:, batch_ind]
                batch_mask = mask[:, batch_ind]

                _, inv_loss, forw_loss = self.icm(batch_actions,
                                                  batch_curr_states,
                                                  batch_next_states,
                                                  batch_mask)
                epoch_forw_loss += forw_loss.item()
                epoch_inv_loss += inv_loss.item()
                unclip_intr_loss = 10 * (0.2 * forw_loss + 0.8 * inv_loss)

                # take gradient step
                self.optimizer_icm.zero_grad()
                unclip_intr_loss.backward()
                self.optimizer_icm.step()
                linear_decay_lr(self.optimizer_icm, self.timestep * 16)
        self.writer.add_scalar('Forward_loss',
                               epoch_forw_loss / (epochs * (len(indexes) // batch_size + 1)),
                               self.timestep
        )
        self. writer.add_scalar('Inv_loss',
                                epoch_inv_loss / (epochs * (len(indexes) // batch_size + 1)),
                                self.timestep
        )
