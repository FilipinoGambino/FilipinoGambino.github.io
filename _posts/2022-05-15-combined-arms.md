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

# Environment Synopsis:

2 teams composed of 45 melee units and 35 ranged units battle against each other. Melee units have a shorter range of both attack and movement, but have more health than their ranged coutnerparts; the agents also regenerate a small amount of their missing health each time step. For now I'll be using the default parameters with the exception of minimap_mode.

# Imports:
```python
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import os
import time

import supersuit as ss
from pettingzoo.magent import combined_arms_v6
import gym

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback
```

# Building the environment

```python
def make_env(fname):
    env = combined_arms_v6.parallel_env(minimap_mode=True)
    env = ss.black_death_v3(env)
    env = ss.agent_indicator_v0(env, type_only=True)
    # env = ss.frame_skip_v0(env, (1,3))
    env = ss.pad_action_space_v0(env)
    env = ss.sticky_actions_v0(env, 0.3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=2, base_class="gym")
    env.is_vector_env = True
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = VecMonitor(env, filename=fname)
    return env
```
1. The 'black_death' wrapper removes deceased agents from the environment.
2. The 'agent_indicator' wrapper adds one layer to the observation space for each type of agent. In this case there is a friendly ranged, friendly melee, enemy range, enemy melee.
3. The 'frame_skip' wrapper allows the environment to repeat actions, ignoring the state and summing the reward.
4. The 'sticky_actions' wrapper gives a probability to repeat actions without ignoring the state. Probably use one or the other. I'll use frame_skip for now.
5. The 'pad_action_space' wrapper pads the melee units action space to match that of the ranged agents.
6. The 'pettingzoo_env_to_vec_env' wrapper makes a vector environment where there is one vector representing each agent. In this case we start with 162 vectors.
7. The 'concat_vec_envs' wrapper concatenates all of the vector environments which will be passed through the model.

# Modeling
```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    batch_size=256,
    seed=42,
)
```
To establish the baseline I'll be using the stable baseline's default model.


TODO:
Finish this post
