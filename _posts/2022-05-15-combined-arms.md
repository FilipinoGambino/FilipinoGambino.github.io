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
<br /><br /><br />

## Environment Synopsis:

2 teams composed of 45 melee units and 36 ranged units battle against each other. Melee units have a shorter range of both attack and movement, but have more health than their ranged coutnerparts; the agents also regenerate a small amount of their missing health each time step. For now I'll be using the default parameters with the exception of minimap_mode.

## Imports:
```python
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

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
    env = VecMonitor(env, filename=fname)
    return env
```
### Supersuit Wrappers
* The 'black_death' wrapper removes deceased agents from the environment.
* The 'agent_indicator' wrapper adds one layer to the observation space for each type of agent. In this case there is a friendly ranged, friendly melee, enemy range, enemy melee.
* The 'frame_skip' wrapper allows the environment to repeat actions, ignoring the state and summing the reward.
* The 'sticky_actions' wrapper gives a probability to repeat actions without ignoring the state. Probably use one or the other. I'll use frame_skip for now.
* The 'pad_action_space' wrapper pads the melee units action space to match that of the ranged agents.
* The 'pettingzoo_env_to_vec_env' wrapper makes a vector environment where there is one vector representing each agent. Since we have 162 agents `(45 + 36) * 2` we get 162 vectors.
* The 'concat_vec_envs' wrapper concatenates all of the vector environments which will be passed through the model.

The 'VecMonitor' tracks the reward, length, time, etc. and saves it the the *filename* log file. 

## Modeling
```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    seed=42,
)
```
To establish the baseline I'll be using the stable baseline's default architecture.


TODO:
Finish this post
