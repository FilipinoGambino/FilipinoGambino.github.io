---
title: "Combined Arms"
categories:
  - Paper Implementations
tags:
  - PettingZoo
  - Combined Arms
  - Multi-Agent
  - RL
  - Stable Baselines3
  - PPO
  - Supersuit
  - WandB
---

Hello and welcome to my first blog post! This is the start of a series of blog posts where I'm going to be working with the multi-agent [Combined Arms](https://www.pettingzoo.ml/magent/combined_arms) environment on [PettingZoo](https://www.pettingzoo.ml/#) and I'll keep adding parts, building it up, testing things out, and later implement a couple of papers I think are interesting.
<br /><br />

## Environment Synopsis:

2 teams composed of 45 melee units and 36 ranged units battle against each other. Melee units have a shorter range of both attack and movement, but have more health than their ranged coutnerparts; the agents also regenerate a small amount of their missing health each time step. For now I'll be using the default parameters with the exception of minimap_mode.
<br /><br />
## Imports:
```python
import numpy as np
import os
import time

import gym
import supersuit as ss
from pettingzoo.magent import combined_arms_v6

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback
```
<br /><br />
## Building the environment

```python
def make_env(fname):
    env = combined_arms_v6.parallel_env(minimap_mode=True)
    env = ss.black_death_v3(env)
    env = ss.agent_indicator_v0(env, type_only=True)
    # env = ss.frame_skip_v0(env, (1,3))
    env = ss.sticky_actions_v0(env, 0.3)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=2, base_class="gym")
    env = gym.wrappers.RecordVideo(env, config.video_logs)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=2048)
    env = VecMonitor(env, filename=fname)
    return env
```
### Supersuit Wrappers
* The 'black_death' wrapper removes deceased agents from the environment.
* The 'agent_indicator' wrapper adds one layer to the observation space for each type of agent. In this case there is a friendly ranged, friendly melee, enemy range, enemy melee.
* The 'frame_skip' wrapper allows the environment to repeat actions, ignoring the state and summing the reward.
* The 'sticky_actions' wrapper gives a probability to repeat actions without ignoring the state. Probably use one or the other. I'll use frame_skip for now.
* The 'pad_action_space' wrapper pads the melee units action space to match that of the ranged agents.
* The 'pettingzoo_env_to_vec_env' wrapper makes a vector environment where there is one vector representing each agent. Since we have 162 agents `(45 ranged + 36 melee) * 2 teams` we get 162 vectors.
* The 'concat_vec_envs' wrapper concatenates all of the vector environments which will be passed through the model.

### Other Wrappers
* The 'VecMonitor' tracks the reward, length, time, etc. and saves it the the *filename* log file. 
* Gym's 'RecordVideo' wrapper will saves videos
* Gym's 'RecordEpisodeStatistics' wrapper records various information from the episode which for us is `162 vectors * 4 environments * 2048 steps` for a total of 1,327,104 total steps per episode.
<br /><br />

## WandB
```python
config = {
    "learning_rate": 3e-4,
    "total_timesteps": int(2e7),
    "log": "/run/ppo",
}

wandb.init(
    config=config,
    name="baseline",
    project="combined_arms_v6",
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
)
```
<br /><br />
## Modeling
```python
env = make_env(config["log"])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=config["learning_rate"],
    tensorboard_log=config["log"],
    seed=42,
)

model.learn(total_timesteps=config.total_timesteps)
wandb.finish()
```
To establish the baseline I'll be using the stable baseline's default architecture which consists of seperate fully connected networks for the policy network and value network.
<br />
<p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_policy_network.jpg" width="350" height="350">
    &emsp;&emsp;&emsp;&emsp;
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_value_network.jpg" width="350" height="350">
</p>


TODO:
Finish this post
