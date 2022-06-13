
---
title: "Another Test"
categories:
  - Petting Zoo's Combined Arms MAgent Environment
tags:
  - PettingZoo
  - Combined Arms
  - MAgent
  - RL
  - SB3
  - PPO
  - Supersuit
  - WandB
---

<style>
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
} 
 
.column {
  float: left;
  width: 50%;
  padding: 5px;
}

.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

Hello and welcome to my first blog post! This is the start of a series of posts where I'm going to be working with the multi-agent [Combined Arms](https://www.pettingzoo.ml/magent/combined_arms) environment of the PettingZoo library and I'll keep adding parts, building it up, testing things out, and later implement a couple of papers I think are interesting.
<br /><br />

## Purpose
The purpose of this series is to showcase a project to potential employers.

## The Environment
There are 2 teams contained in a 45x45 map and each team is composed of 45 melee units and 36 ranged units. Melee units have a shorter range for both attack and movement than their ranged counterparts, but have more health. The units or agents also slowly regenerate a small amount of their missing health since it takes multiple attacks to kill an agent. Agents are rewarded for injurying/killing opposing agents and negatively rewarded for both injuring/killing friendly agents or dying.

<div class="row">
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/combined_arms_v6_opening.png" alt="Starting Position" height=350>
  </div>
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/combined_arms_v6_one_step.png" alt="First Step" height=350>
  </div>
</div>

## Imports
```python
import gym
import supersuit as ss
from pettingzoo.magent import combined_arms_v6

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor

# For logging
import wandb
from wandb.integration.sb3 import WandbCallback
```

## Wrapping the Environment

```python
def make_env(fname):
    env = combined_arms_v6.parallel_env(minimap_mode=True)
    env = ss.black_death_v3(env)
    env = ss.agent_indicator_v0(env, type_only=True)
    # env = ss.frame_skip_v0(env, (1,5))
    env = ss.sticky_actions_v0(env, 0.3)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=2, base_class="gym")
    env = gym.wrappers.RecordVideo(env, "videos/")
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=2048)
    env = VecMonitor(env, filename=fname)
    return env
```

* 'black_death' removes deceased agents from the environment.
* 'agent_indicator' adds one layer to the observation space for each type of agent. In this case we have a friendly ranged, friendly melee, enemy ranged, enemy melee.
* 'frame_skip' allows the environment to repeat actions, ignoring the state and summing the reward.
* 'sticky_actions' gives a probability to repeat actions. I'll use this one over frame_skip for now.
* 'pad_action_space' pads the melee agents action space to match that of the ranged agents.
* 'pettingzoo_env_to_vec_env' makes a vector environment where there is one vector representing each agent. Since we have 162 agents `(45 ranged + 36 melee) * 2 teams` we get 162 vectors per environment.
* 'concat_vec_envs' concatenates all of the vector environments which will be passed through the model.
* 'VecMonitor' tracks the reward, length, time, etc. and saves it the the *filename* log file. 
* Gym's 'RecordVideo' wrapper saves videos
* Gym's 'RecordEpisodeStatistics' wrapper records various information from the episode which for us is `162 vectors * 4 environments * 2048 steps` for a total of 1,327,104 total steps per episode.
<br /><br />

## Learning Rate Scheduler
```python
def lr_scheduler(min_lr: float, max_lr: float, sw_perc: float) -> Callable:
    """
    :param min_lr: Minimum learning rate.
    :param max_lr: Maximum learning rate
    :param sw_perc: Progress interval for piecewise function
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        x = 1 - progress_remaining
        if x < sw_perc:
            return max_lr * (x / sw_perc) + min_lr
        else:
            return max_lr * (max_lr*100) ** ((x - sw_perc) / (1 - sw_perc))

    return func
```
Setting the minimum learning rate to 1e-8, the maximum to 1e-4, and the interval to 0.2 we get the below learning rate schedule.

<div class="center">
  <img src="https://filipinogambino.github.io/ngorichs/assets/images/lr_schedule_plot.jpg">
</div>

## Now we can build our model

To establish the baseline I'll be using sb3's default architecture which consists of seperate fully connected layers for the policy network and value network. Melee agents only have 9 available actions (though we padded their actions spaces to match the ranged agents 25), so hopefully there are enough parameters here to learn which actions do nothing.
<br />

<div class="row">
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_policy_network.jpg" alt="Policy Network">
  </div>
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_value_network.jpg" alt="Value Network">
  </div>
</div>

<br />
Now we write a quick config dictionary since we'll need to pass some of the parameters when we initialize wandb and we can assemble our model.

```python
config = {
    "learning_rate": lr_scheduler(1e-8, 1e-4, 0.2),
    "total_timesteps": int(2e7),
    "log": "/runs/ppo",
}

env = make_env(config["log"])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=config["learning_rate"],
    tensorboard_log=config["log"],
    seed=42,
)
```

## Modeling

Then all we have left to do is initialize our wandb instance for logging and we can start training.

```python
wandb.init(
    config=config,
    name="baseline",
    project="combined_arms_v6",
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
)

model.learn(total_timesteps=config.total_timesteps)
wandb.finish()
```
## Results
<p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_wandb.jpg">
</p>

Unsurprisingly this baseline didn't perform all that well after ~20 million steps. It looks like it could go up a little more with more training, but I think it's time to move on to a different architecture. In the [next post](https://filipinogambino.github.io/ngorichs/blog/combined-arms-part-2/) I'll be building a convolutional neural network along with some embeddings to differentiate the different types of agents.
