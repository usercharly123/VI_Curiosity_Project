import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import Memory
from ICMPPO16 import ICMPPO
import torch.nn as nn
from torch.distributions import Categorical

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_pettingzoo_base_env import UnityPettingzooBaseEnv
from onnx_wrapper import ONNXWrapper

import time

class CustomUnityEnv(UnityPettingzooBaseEnv):
    def __init__(self, env):
        super().__init__(env)
        self.all_dones = np.zeros(16, dtype=bool)
        self.all_rewards = np.full(16, -0.001, dtype=np.float32)
        self.last_state = None  # Store the last valid state
    
    def step(self, actions):
        next_state, rewards, dones, info = super().step(actions)
        
        # Update the done flags for all agents
        if not isinstance(dones, dict):
            # If dones is a single value
            if dones.size == 1 and dones.item():
                # Get the current agent's name from the environment
                current_agent = self._agents[self._agent_index]
                print("AGENT", current_agent, "done:", dones.item())
                print("STATE", next_state.shape, "REWARDS", rewards)
                # Extract agent_id from the name format "Pyramids?team=0?agent_id=X"
                try:
                    agent_id = int(current_agent.split('agent_id=')[1])
                    self.all_dones[agent_id] = True
                    self.all_rewards[agent_id] = np.float32(rewards)
                    # Create a copy of the last valid state
                    full_state = self.last_state.copy()
                    # Update only the state for the non-done agents
                    for i, is_done in enumerate(self.all_dones):
                        if not is_done and i < next_state.shape[0]:
                            full_state[i] = next_state[i]
                    next_state = full_state
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse agent ID from {current_agent}")
        
        else:
            self.all_dones = dones
            self.all_rewards = np.array(rewards, dtype=np.float32)
            self.last_state = next_state.copy()
        
        return next_state, self.all_rewards, self.all_dones, info
    
    def reset(self):
        state = super().reset()
        self.all_dones = np.zeros(16, dtype=bool)
        self.all_rewards = np.full(16, -0.001, dtype=np.float32)
        self.last_state = state.copy()  # Store the initial state
        return state

def main():
    
    parser = argparse.ArgumentParser(description="Train agent in Unity environment")
    parser.add_argument("--max-episodes", type=int, default=50, help="Maximum number of training episodes")
    parser.add_argument("--update-timestep", type=int, default=2048, help="Update policy every n timesteps")
    parser.add_argument("--os", type=str, choices=["linux", "windows"], required=True, help="Operating system (linux or windows)", default="linux")
    parser.add_argument("--reward_mode", type=str, choices=["intrinsic", "extrinsic", "both"], default="both", help="Reward mode to use (intrinsic, extrinsic, or both)")
    parser.add_argument("--graphics", action='store_true', help="Whether to visualize the agent during training")
    parser.add_argument("--load_model", action='store_true', help="Whether to load the last saved model")
    parser.add_argument("--permute", action='store_true', help="Whether to permute the state space of the agent")
    parser.add_argument("--perturb", action='store_true', help="Whether to perturb the weights of the policy with Gaussian noise")
    args = parser.parse_args()

    current_os = args.os

    if current_os == "linux":
        env_path = "Pyramid/Pyramids16_linux_half_agents/Pyramids16_linux_half_agents.x86_64"
        # Set LD_LIBRARY_PATH for Linux
        mono_path = os.path.join(os.path.dirname(env_path), "Pyramids16_linux_half_agents_Data/MonoBleedingEdge/x86_64")
        os.environ["LD_LIBRARY_PATH"] = mono_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    elif current_os == "windows":
        env_path = 'Pyramid/Pyramids16_windows_half_agents'

    solved_reward = 1000     # stop training if avg_reward > solved_reward
    log_interval = 100     # print avg reward in the interval
    max_episodes = args.max_episodes  # WAS 350      # max training episodes
    max_timesteps = 1000    # WAS 1000 max timesteps in one episode
    update_timestep = args.update_timestep  # update policy every n timesteps
    
    print(f"Running on {current_os} for {max_episodes} episodes...")

    # Initialize Unity env
    unity_env = UnityEnvironment(env_path, no_graphics=not args.graphics, seed=42)
    multi_env = CustomUnityEnv(unity_env)

    # Choose the reward mode
    reward_mode = args.reward_mode
    # Initialize the log directory to events/reward_mode
    log_dir = 'Pyramid/events/' + reward_mode + '/'

    # Initialize log_writer, memory buffer, icmppo
    writer = SummaryWriter(log_dir)
    memory = Memory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = ICMPPO(writer=writer, device=device, reward_mode=reward_mode, lr=3e-4)

    # Path to the saved models
    model_dir = os.path.join('Pyramid', 'models', reward_mode)
    ppo_path = os.path.join(model_dir, 'ppo100.pt')
    icm_path = os.path.join(model_dir, 'icm100.pt')

    # Load the last saved policy and ICM if they exist
    load_model = args.load_model
    permute = args.permute
    perturb = args.perturb

    if load_model:
        if os.path.exists(ppo_path):
            print(f"Loading policy weights from {ppo_path}")
            # Load weights into both networks
            state_dict = torch.load(ppo_path, map_location=device)
            agent.policy.load_state_dict(state_dict)
            agent.policy_old.load_state_dict(state_dict)
            
            # Perturb the weights with Gaussian noise if needed
            if perturb:
                print("Perturbing weights with Gaussian noise...")
                # Set random seed for reproducibility of perturbations
                torch.manual_seed(42)
                
                # Create perturbations once and apply to both networks
                perturbations = {}
                for name, param in agent.policy.named_parameters():
                    perturbations[name] = torch.randn_like(param) * 0.17
                
                # Apply the same perturbations to both networks
                with torch.no_grad():
                    for name, param in agent.policy.named_parameters():
                        param.add_(perturbations[name])
                    for name, param in agent.policy_old.named_parameters():
                        param.add_(perturbations[name])
                
                print("Applied identical perturbations to both networks")
            
        if os.path.exists(icm_path):
            print(f"Loading ICM weights from {icm_path}")
            agent.icm.load_state_dict(torch.load(icm_path, map_location=device))

    timestep = 0
    T = np.zeros(16)
    state = multi_env.reset()
    print(state.shape)

    # Initialize the time to know how ong the script takes to run
    start_time = time.time()
    print("Start time: ", start_time)

    # training loop
    for i_episode in range(1, max_episodes + 1):
        # logging the results from the previous episode
        if i_episode > 1:
            print('Episode {} \t episode reward: {} \t'.format(i_episode, episode_rewards.mean()))
            writer.add_scalar('Mean_extr_reward_per_1000_steps',
                            episode_rewards.mean(),
                            timestep
            )
            
        print("Episode: ", i_episode)
        episode_rewards = np.zeros(16)
        
        # Initialize dones array
        dones = np.zeros(16, dtype=bool)
        
        for i in range(max_timesteps):
            timestep += 1
            T += 1
            # Running policy_old:
            # Turn the states of type numpy_object_ to numpy array
            actions = agent.policy_old.act(np.array(state), memory, permute=permute)
            
            # Zero out actions for done agents
            actions = np.array(actions)
            actions[dones] = 0  # Set actions to 0 for done agents
            
            state, rewards, dones, info = multi_env.step(list(actions))
            rewards = np.array(rewards)
            dones = np.array(dones, dtype=bool)
            
            # Handle rewards differently based on OS
            if current_os == "linux":
                # On Linux, ensure rewards are 1D array
                if rewards.ndim == 2:
                    rewards = rewards.flatten()
            
            T[dones] = 0
            # Saving reward and is_terminal:
            memory.rewards.append(rewards)
            memory.is_terminals.append(dones)
            
            # If any agent is done, reset the environment
            if dones.any():
                print(f"Episode {i_episode} done at timestep {i}")
                # print the rewards of the agents that are done
                print("Rewards: ", rewards)
                print("Dones: ", dones)
                state = multi_env.reset()  # Reset the environment when done = True
                break  # End the episode loop

            episode_rewards += rewards
            
            if timestep % 100 == 0:
                print("Timestep: ", timestep)
                print("Episode rewards: ", episode_rewards)

            # Update policy every 2048 timesteps
            if timestep % update_timestep == 0 and len(memory.states) > 0:
                print(f"Updating policy at timestep {timestep}")
                agent.update(memory, timestep, current_os=current_os)
                memory.clear_memory()  # Clear memory after update        
            
    # Save in models/reward_mode
    dir = 'Pyramid/models/' + reward_mode + '/'
    if permute:
        dir = 'Pyramid/models/' + reward_mode + '_permute/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(agent.policy_old.state_dict(), dir + 'ppo.pt')
    torch.save(agent.icm.state_dict(), dir + 'icm.pt')

    # Export policy to ONNX for Unity inference
    # Instantiate your wrapper
    onnx_model = ONNXWrapper(agent.policy_old)
    onnx_model.cpu()  # Move model to CPU
    onnx_path = dir + 'Pyramid.onnx'

    # Prepare dummy inputs
    dummy_obs_0 = torch.randn(1, 56)
    dummy_obs_1 = torch.randn(1, 56)
    dummy_obs_2 = torch.randn(1, 56)
    dummy_obs_3 = torch.randn(1, 4)
    dummy_action_masks = torch.ones(1, 5)  # Shape: [batch, num_actions], all ones (no masking)

    # Export
    torch.onnx.export(
        onnx_model,
        (dummy_obs_0, dummy_obs_1, dummy_obs_2, dummy_obs_3, dummy_action_masks),
        onnx_path,
        input_names=['obs_0', 'obs_1', 'obs_2', 'obs_3', 'action_masks'],
        output_names=['discrete_actions'],
        opset_version=11,
        dynamic_axes={
            'obs_0': {0: 'batch_size'},
            'obs_1': {0: 'batch_size'},
            'obs_2': {0: 'batch_size'},
            'obs_3': {0: 'batch_size'},
            'action_masks': {0: 'batch_size'},
            'discrete_actions': {0: 'batch_size'}
        }
    )
    print(f"Exported ONNX model to {onnx_path}")
            
    # Print the time it took to run the script
    end_time = time.time()
    print("End time: ", end_time)
    print("Time taken: ", end_time - start_time)

    multi_env.close()   # Closes the UnityToGymWrapper
    writer.close()      # Closes TensorBoard logging


if __name__ == "__main__":
    main()