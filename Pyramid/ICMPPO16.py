import numpy as np
import torch
import torch.nn as nn
from ICM16 import ICM
from ActorCritic import ActorCritic
from utils import Swish, linear_decay_beta, linear_decay_lr, linear_decay_eps


class ICMPPO:
    def __init__(self, writer, state_dim=172, action_dim=5, n_latent_var=512, lr=3e-4, betas=(0.9, 0.999),
                 gamma=0.99, ppo_epochs=3, icm_epochs=1, eps_clip=0.2, ppo_batch_size=128,
                 icm_batch_size=16, intr_reward_strength=0.02, lamb=0.95, device='cpu', reward_mode='both', 
                 decaying_lr=False, max_episodes=350, last_epoch=1):
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

        # Create policy network
        self.policy = ActorCritic(state_dim=state_dim,
                                  action_dim=action_dim,
                                  n_latent_var=n_latent_var,
                                  activation=Swish(),
                                  device=self.device,
                                  writer=self.writer
                                  ).to(self.device)
        
        # Create old policy network
        self.policy_old = ActorCritic(state_dim,
                                      action_dim,
                                      n_latent_var,
                                      activation=Swish(),
                                      device=self.device
                                      ).to(self.device)

        # Copy weights from policy to policy_old
        state_dict = self.policy.state_dict()
        self.policy_old.load_state_dict(state_dict)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr}
        ])
        
        # Initialize learning rate scheduler with warm-up and cosine decay
        def lr_lambda(step):
            # Warm-up for first 10% of training
            warmup_steps = max_episodes * 0.1  # 10% of total episodes
            if step < warmup_steps:
                return step / warmup_steps
            # Cosine decay for remaining 90%
            progress = (step - warmup_steps) / (max_episodes - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))  # Decays from 1 to 0.5
        
        if decaying_lr:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR( 
                self.optimizer,
                lr_lambda,
                last_epoch=last_epoch-1  # -1 because PyTorch increments before computing lr
            )
        else:
            self.scheduler = None
        
        self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss(reduction='none')

    def update(self, memory, timestep, current_os="linux"):
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
        
        rewards_np = np.array(memory.rewards[:-1])  # each reward is a list of 16 elements so rewards_np is (N, 16) 
        rewards = torch.tensor(rewards_np).T.to(self.device).detach()
        
        # print the total rewards of each agent
        total_extr_rewards = rewards.sum(dim=1)
        print("Extrinsic rewards shape:", rewards.shape)
        print("total extrinsic reward shape:", total_extr_rewards.shape)
        print(f"Total extrinsic rewards for each agent: {total_extr_rewards.cpu().numpy()}")
        
        mask = (~torch.tensor(
            np.array(memory.is_terminals),
            dtype=torch.bool
        ).T[:, :-1].to(self.device)).long()
        
        with torch.no_grad():
            intr_reward, _, _ = self.icm(actions, curr_states, next_states, mask)
        intr_rewards = torch.clamp(self.intr_reward_strength * intr_reward, 0, 1)
        total_intr_rewards = intr_rewards.sum(dim=1)
        print("Intrinsic rewards shape:",intr_rewards.shape)
        print("Total intrinsic rewards shape:", total_intr_rewards.shape)
        print(f"Total intrinsic rewards for each agent: {0.1*total_intr_rewards.cpu().numpy()}")

        # Combine rewards based on reward_mode
        if self.reward_mode == 'both':
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
            state_values = torch.squeeze(self.policy.value_layer(curr_states))
            next_state_values = torch.squeeze(self.policy.value_layer(next_states))
            td_target = combined_rewards + self.gamma * next_state_values * mask
            delta = td_target - state_values
            print("Delta shape:", delta.shape)

            self.writer.add_scalar('maxValue',
                                   state_values.max(),
                                   timestep
                                   )
            self.writer.add_scalar('meanValue',
                                   state_values.mean(),
                                   self.timestep
                                   )

            advantage = torch.zeros(1, 16).to(self.device)      
            advantage_lst = []
            for i in range(delta.size(1) - 1, -1, -1):
                delta_t, mask_t = delta[:, i], mask[:, i]
                advantage = delta_t + (self.gamma * self.lambd * advantage) * mask_t
                advantage_lst.insert(0, advantage)

            advantage_lst = torch.cat(advantage_lst, dim=0).T
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
                if self.scheduler is not None:
                    # Update learning rate
                    self.scheduler.step()

                epoch_surr_loss += loss.item()

        self._icm_update(self.icm_epochs, self.icm_batch_size, curr_states, next_states, actions, mask)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, self.timestep)
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

        self.writer.add_scalar('Forward_loss',
                               epoch_forw_loss / (epochs * (len(indexes) // batch_size + 1)),
                               self.timestep
        )
        self. writer.add_scalar('Inv_loss',
                                epoch_inv_loss / (epochs * (len(indexes) // batch_size + 1)),
                                self.timestep
        )
