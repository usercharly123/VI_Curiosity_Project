import numpy as np
import yaml
import argparse
from IPython.display import HTML
from base64 import b64encode
import cv2
import os, random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

from torch.multiprocessing import Process

import gym_super_mario_bros
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.envs import *


def create_environment():
  en = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', new_step_api=True)
  en = JoypadSpace(en, SIMPLE_MOVEMENT)
  en = SkipFrame(en, skip = 4)
  en = CustomReward(en)
  en = CutGrayScaleObservation(en)
  en = ResizeObservation(en, shape = 84)
  en = FrameStack(en, num_stack = 4)
  return en



def compute_target_advantage(reward, done, value, gamma, n_step, num_workers):
    discounted_return = np.empty([num_workers, n_step])

    # Take the value from the last state (s_{t_max})
    running_add = value[:, -1]
    for t in range(n_step - 1, -1, -1):
        running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
        discounted_return[:, t] = running_add

    # For Actor
    adv = discounted_return - value[:, :-1]

    return discounted_return.reshape(-1), adv.reshape(-1)


def create_video(images, file_name="output"):
    # Remove old videos
    if os.path.exists(f"./video/{file_name}.mp4"):
        os.remove(f"./video/{file_name}.mp4")
    if os.path.exists(f"./video/{file_name}_compressed.mp4"):
        os.remove(f"./video/{file_name}_compressed.mp4")
    
    # set the fourcc codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # get the height and width of the first image
    height, width, _ = images[0].shape

    # create a VideoWriter object
    fps = 20
    out = cv2.VideoWriter(
        f"./video/{file_name}.mp4", fourcc, float(fps), (width, height))

    # write each image to the video file
    for img in images:
        # convert image to BGR color space
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    # release the VideoWriter object
    out.release()

    # Compressed video path
    compressed_path = f"./video/{file_name}_compressed.mp4"
    os.system(
        f"ffmpeg -i ./video/{file_name}.mp4 -vcodec libx264 {compressed_path}")

def show_video(compressed_path="./video/output_compressed.mp4"):
    mp4 = open(compressed_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
  <video width=400 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url)

def plot_rewards(extrinsic, path, window=50):
    steps = list(range(len(extrinsic)))

    # Compute moving average
    moving_avg = pd.Series(extrinsic).rolling(window=window).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(steps, extrinsic, alpha=0.3, label='Reward')
    plt.plot(steps, moving_avg, linewidth=2, label=f'Moving Average ({window}-step)')

    plt.title('Training Rewards and Moving Average Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)


def save_rewards(extrinsic, intrinsic, loss, path):
    with open(path, 'wb') as f:
        np.save(f, extrinsic)
        np.save(f, intrinsic)
        np.save(f, loss)

def load_rewards(path):
    with open(path, 'rb') as f:
        extrinsic = np.load(f)
        intrinsic = np.load(f)
        loss = np.load(f)
    return extrinsic, intrinsic, loss
    
def standardize_array(array, mean, std):
    return (array - mean) / (std + 1e-8)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--init_model', type=str, default=None)
    parser.add_argument('--init_icm', type=str, default=None)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--curiosity', action='store_true')
    parser.add_argument('--extrinsic', action='store_true')
    parser.add_argument('--perturb', action='store_true')

    parser.add_argument('--global_epochs', type=int, default=100000)
    parser.add_argument('--tr_epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_step', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)


    return parser.parse_args()

def device_setup(seed = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    return device

def invert_leftright(action): 
    new_actions = []
    for button in action:
        if (button == 'left'): new_actions.append('right')
        elif (button == 'right'): new_actions.append('left')
        else: new_actions.append(button)
    return new_actions
