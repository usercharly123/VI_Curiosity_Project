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

def main():
    
    parser = argparse.ArgumentParser(description="Train agent in Unity environment")
    parser.add_argument("--max-episodes", type=int, default=50, help="Maximum number of training episodes")
    parser.add_argument("--os", type=str, choices=["linux", "windows"], required=True, help="Operating system (linux or windows)", default="linux")
    parser.add_argument("--reward_mode", type=str, choices=["intrinsic", "extrinsic", "both"], default="both", help="Reward mode to use (intrinsic, extrinsic, or both)")
    parser.add_argument("--graphics", action='store_true', help="Whether to visualize the agent during training")
    parser.add_argument("--load_model", action='store_true', help="Whether to load the last saved model")
    parser.add_argument("--permute", action='store_true', help="Whether to permute the state space of the agent")
    args = parser.parse_args()

    current_os = args.os

    if current_os == "linux":
        env_path = "Pyramid/Pyramid_linux_half/Pyramid_linux_half.x86_64"
    elif current_os == "windows":
        env_path = 'Pyramid/Pyramids16half_agents'

    solved_reward = 1000     # stop training if avg_reward > solved_reward
    log_interval = 100     # print avg reward in the interval
    max_episodes = args.max_episodes  # WAS 350      # max training episodes
    max_timesteps = 1000    # WAS 1000 max timesteps in one episode
    update_timestep = 1024  # WAS 2048 Replay buffer size, update policy every n timesteps
    
    print(f"Running on {current_os} for {max_episodes} episodes...")

    # Initialize Unity env
    unity_env = UnityEnvironment(env_path, no_graphics=not args.graphics, seed=42)
    multi_env = UnityPettingzooBaseEnv(unity_env)

    # Choose the reward mode
    reward_mode = args.reward_mode
    # Initialize the log directory to events/reward_mode
    log_dir = 'Pyramid/events/' + reward_mode + '/'

    # Initialize log_writer, memory buffer, icmppo

    writer = SummaryWriter(log_dir)
    memory = Memory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = ICMPPO(writer=writer, device=device, reward_mode=reward_mode)

    # Path to the saved models
    model_dir = f'Pyramid/models/'
    ppo_path = os.path.join(model_dir, 'ppo_orig.pt')
    icm_path = os.path.join(model_dir, 'icm.pt')

    # Load the last saved policy and ICM if they exist
    load_model = args.load_model
    permute = args.permute
    perturb = False

    if load_model:
        if os.path.exists(ppo_path):
            print(f"Loading policy weights from {ppo_path}")
            agent.policy_old.load_state_dict(torch.load(ppo_path, map_location=device))
            # Perturb the weights with Gaussian noise
            def perturb_weights(model, std=0.175):
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * std)
            if perturb:
                perturb_weights(agent.policy_old, std=0.17)
                # Use the same perturbation for the policy
                agent.policy.load_state_dict(agent.policy_old.state_dict())
                print("Perturbed the weights of the policy with Gaussian noise")
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
        print("Episode: ", i_episode)
        episode_rewards = np.zeros(16)
        episode_counter = np.zeros(16)
        
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
            
            # Ensure dones is a boolean array with exactly 16 elements
            dones = np.array(dones, dtype=bool)
            if dones.size == 1:  # If dones is a scalar or single-element array
                dones = np.full(16, dones.item(), dtype=bool)
            elif dones.size != 16:  # If dones has wrong size
                print(f"Warning: dones has unexpected size {dones.size}, reshaping to 16")
                dones = np.resize(dones, 16)
            
            episode_counter += dones
            T[dones] = 0
            # Saving reward and is_terminal:
            memory.rewards.append(rewards)
            memory.is_terminals.append(dones)
            
            # If any agent is done, reset the environment
            if dones.any():
                print(f"Episode {i_episode} done at timestep {i}")
                state = multi_env.reset()  # Reset the environment when done = True
                break  # End the episode loop

            episode_rewards += rewards
            
            if timestep % 50 == 0:
                print("Timestep: ", timestep)
                print("Rewards: ", episode_rewards)

            # Update policy every 2048 timesteps
            if timestep % 2048 == 0 and len(memory.states) > 0:
                print(f"Updating policy at timestep {timestep}")
                agent.update(memory, timestep)
                memory.clear_memory()  # Clear memory after update

        if episode_counter.sum() == 0:
            episode_counter = np.ones(16)

        # stop training if avg_reward > solved_reward
        if episode_rewards.sum() / episode_counter.sum() > solved_reward:
            print("########## Solved! ##########")
            print("Average reward: ", episode_rewards.sum() / episode_counter.sum())
            writer.add_scalar('Mean_extr_reward_per_1000_steps',
                            episode_rewards.sum() / episode_counter.sum(),
                            timestep
            )
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
            break

        # logging
        if timestep % log_interval == 0:
            print('Episode {} \t episode reward: {} \t'.format(i_episode, episode_rewards.sum() / episode_counter.sum()))
            writer.add_scalar('Mean_extr_reward_per_1000_steps',
                            episode_rewards.sum() / episode_counter.sum(),
                            timestep
            )
            
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