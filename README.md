# CS503 VI_Curiosity_Project

This repository contains our implementation of assessing the properties of Pathak et al's definition of curiosity under permutation of state space in two environments:
- the Pyramid game from the Unity environment
- the Mario game from the Gymnasium environment

## **Installation**  
To begin the experiments, you first need to install the required packages and dependencies. To do this, please run the [setup_env.sh](setup_env.sh) script.

```bash
bash setup_env.sh
```

## **Pyramid environment**
The Pyramid environment consists of a square 2D environment where the blue agent needs to press a button to spawn a pyramid, then navigate to the pyramid, knock it over, and move to the gold brick at the top to get a reward.

### Building the profiles for the environment (optional)
To work in the Pyramid environment, you can **either** take the original environment from Unity:
- download the Unity Game Engine ([Unity Hub]([url](https://unity.com/fr/download)))
- in the Unity Game Engine, download the recommanded editor version 6000.1.1f1
- clone the [ml-agent github]([url](https://github.com/Unity-Technologies/ml-agents)) with
```bash
git clone https://github.com/Unity-Technologies/ml-agents.git
```
- create a new Project in the Unity Editor (v6000.1.1f1), and open the project
- go to the top menu and select Window -> Package Management -> Package Manager
- in the top left-hand corner of the window that opens, select "+", "Install package from disk", and select your local path to the package of the ml-agents library (should be in ml-agents/Project/Assets/ML-Agents/)
- close the Package Manager window. Now at the bottom of your Project Window the ml-agents files should appear. Select "Assets/ML-Agents/Examples/Pyramids"
- Under Pyramids, select Scenes and double click on the Pyramid scene, which should open the scene
- under File/Build Profiles, at the top left-hand corner of the editor, you can build profiles for your Windows/Mac/Linux/... operating system

**OR**, you can just take our pre-exported profiles in the Pyramid16half_agents_windows or Pyramid16half_agents_linux folders. Note that the linux version has only bee, tested on the SCITAS cluster, and has been built for linux servers, which may be different from your linux machine.

### Training the agent
