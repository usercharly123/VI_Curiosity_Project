import numpy as np
import random
import time
import torch
import threading
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.agent import *
from src.envs import *
from src.utils import *

import argparse
import pathlib

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

num_workers = 8
envs = [create_environment() for _ in range(num_workers)]
data = [None for _ in range(num_workers)]

def thread_routine(idx, action):
    global envs
    env = envs[idx]
    state, reward, done, info = env.step(action)

    if done:
        state = env.reset()
    global data
    data[idx] = [state, reward, done]

def send_actions(n, actions):
    threads = []
    for i in range(n):
        t = threading.Thread(target = thread_routine, args = (i, actions[i]))
        t.start()
        threads.append((i,t))
    return threads


def main():

    args = parse_args()

    path = args.results_path
    model_path = args.init_model
    icm_path = args.init_icm
    use_curiosity = args.curiosity
    use_extrinsic = args.extrinsic
    pertubation = args.perturb

    global_epochs = args.global_epochs
    training_epochs = args.tr_epochs
    batch_size = args.batch_size
    learning_rate = args.lr

    n_step = args.n_step
    gamma = 0.99
    hidden_size = 512

    device = device_setup()

    env = create_environment()
    n_actions = env.action_space.n

    agent = Agent(n_actions=n_actions, device=device,
                    use_icm=use_curiosity, batch_size=batch_size, 
                    hidden_size=hidden_size, learning_rate=learning_rate,
                    epochs=training_epochs)

    if model_path is not None: 
        agent.model.load_state_dict(torch.load(model_path, weights_only=True))

    if icm_path is not None:                        
        agent.icm.load_state_dict(torch.load(icm_path, weights_only=True))
        
    ext_reward = []
    int_reward = []
    losses = []

    global envs
    for environ in envs:
        environ.reset()
    
    state = env.reset()
    states = [state for _ in range(num_workers)]

    for epoch in range(global_epochs):

        t0 = time.time()

        h_states, h_extrinsic_r, h_intrinsic_r, h_dones, h_next_states, h_actions, h_values, h_policies = [], [], [], [], [], [], [], []

        global data
        data = [None for _ in range(num_workers)]

        for step in range(n_step):

            actions, value, policy = agent.sample_action(states)

            if pertubation: 
                actions = invert_leftright(actions)

            workers = send_actions(num_workers, actions)

            next_states = [None for _ in range(num_workers)]
            rewards = [None for _ in range(num_workers)] 
            dones = [None for _ in range(num_workers)] 

            for i, w in workers:
                w.join()
                ns, r, d = data[i]
                next_states[i] = ns
                rewards[i] = r
                dones[i] = d

            ext_rewards = np.array(rewards)
            dones = np.array(dones)

            int_rewards = agent.intrinsic_reward(
                states, 
                next_states,
                actions)
            
            h_intrinsic_r.append(int_rewards)
            h_states.append(states)
            h_next_states.append(next_states)
            h_extrinsic_r.append(ext_rewards)
            h_dones.append(dones)
            h_actions.append(actions)
            h_values.append(value)
            h_policies.append(policy)

            states = next_states.copy()

            ext_reward.append(ext_rewards.mean())
            int_reward.append(int_rewards.mean())

        _, value, _ = agent.sample_action(states)
        h_values.append(value)

        total_state = np.stack(h_states).transpose(
            [1, 0, 2, 3, 4]).reshape(-1, *state.shape)
        total_next_state = np.stack(h_next_states).transpose(
            [1, 0, 2, 3, 4]).reshape(-1, *state.shape)

        # total_intrinsic_r: [num_workers, n_step]
        total_intrinsic_r = np.stack(h_intrinsic_r).T 
        total_extrinsic_r = np.stack(h_extrinsic_r).T
        total_extrinsic_r *= 1e-1

        total_action = np.stack(h_actions).T.reshape(-1)
        total_value = np.stack(h_values).T
        total_done = np.stack(h_dones).T

        total_r = np.zeros_like(total_extrinsic_r)

        if use_extrinsic: 
            total_r += total_extrinsic_r

        if use_curiosity: 
            total_r += total_intrinsic_r

        target, advantage = compute_target_advantage(
            total_r,
            total_done,
            total_value,
            gamma,
            n_step,
            num_workers
        )

        advantage = standardize_array(advantage, np.mean(advantage), np.std(advantage))
        target = standardize_array(target, np.mean(target), np.std(target))

        loss = agent.train(total_state,
                           total_next_state,
                           target,
                           total_action,
                           advantage,
                           h_policies)
        
        losses.append(loss)

        print(f"\nepoch {epoch + 1}: loss = {np.mean(losses):.3e}, reward = {np.mean(ext_reward):.3f}, curiosity = {np.mean(int_reward):.3e}\n, time = {time.time() - t0}")

        if (epoch % 100 == 0) or (epoch == global_epochs - 1):

            if agent.use_icm: 

                name_icm = f"results/models/ICM_extr{use_extrinsic}_perturbation-{pertubation}_train-{training_epochs}.pt"
                torch.save(agent.icm.state_dict(), path + name_icm)

            name_model = f"results/models/model_curiosity-{use_curiosity}_extr{use_extrinsic}_perturbation-{pertubation}_train-{training_epochs}.pt"
            torch.save(agent.model.state_dict(), path + name_model)

            numpy_path = f"results/arrays/model_curiosity-{use_curiosity}_extr{use_extrinsic}_perturbation-{pertubation}_train-{training_epochs}.npy"
            plot_path = f"results/plots/model_curiosity-{use_curiosity}_extr{use_extrinsic}_perturbation-{pertubation}_train-{training_epochs}.png"
            save_rewards(ext_reward, int_reward, losses, path + numpy_path)
            plot_rewards(ext_reward, path + plot_path, window = 50)


if __name__ == '__main__':
    main()

