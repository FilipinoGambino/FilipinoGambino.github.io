---
title: "Baseline"
permalink: /:categories/archive/baseline/
author_profile: false
sidebar:
  nav: "combined_arms"
categories:
  - combined_arms
tags:
  - PettingZoo
  - RL
  - SB3
  - PPO
  - WandB
---

<style>
.centertext {text-align: center;}

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

Firstly, we need to establish a baseline to compare our future models to; we need to know that whatever modifications we make are actually useful in improving the agents' decisions. Now, allowing the agents to take completely random actions would certainly be a baseline, but the teams aren't evenly distributed on the map and killing an agent requires several attacks in a few time steps. This would require at least a little coordination. Instead we'll train a small, fully connected network which is the default model in SB3's policy algorithms like PPO. We'll also just use the default parameters for the algorithm with a few exceptions like using a learning rate scheduler instead of the default constant learning rate.

## Imports
To get started, we'll do our imports. Note the WandB API key to connect to your WandB account.

```python
# Creating the environment
from pettingzoo.magent import combined_arms_v6

# Wrapping the environment
import supersuit as ss
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor

# For the policy algorithm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

# For logging on Weights and Biases
import wandb
from wandb.integration.sb3 import WandbCallback

# For WandB API key
import os

os.environ["WANDB_API_KEY"]='YOUR_API_KEY_HERE'
```

## Wrapping the Environment
The environment is another place we'll be straying from the default values. This includes *minimap_mode* which adds 6 layers containing agent locations to the observation space and the *agent_indicator* wrapper with *type_only* set to "True" which adds a sort of one-hot encoding set of layers to the observation space for each of the 4 types of agents (friendly ranged, friendly melee, enemy ranged, enemy melee).

```python
def make_env():
  """
  :return: a number of wrapped, parallel, combined arms environments ready for logging to Weights and Biases
  """
    env = combined_arms_v6.parallel_env(max_cycles=2000, minimap_mode=True)
    env = ss.pad_action_space_v0(env)
    env = ss.black_death_v3(env)
    env = ss.agent_indicator_v0(env, type_only=True)
    # env = ss.frame_skip_v0(env, (1,3))
    env = ss.sticky_actions_v0(env, 0.3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=2, base_class="gym")
    env = VecMonitor(env, filename=None)
    env = VecVideoRecorder(
        env,
        "videos/",
        record_video_trigger=lambda x: x % 2048 == 0,
        video_length=500,
    )
    return env
```

* 'pad_action_space' pads the melee agents action space (Discrete 9) to match that of the ranged agents (Discrete 25).
* 'black_death' removes deceased agents from the environment.
* 'agent_indicator' adds one layer to the observation space for each type of agent. In this case we have a friendly ranged, friendly melee, enemy ranged, enemy melee.
* 'frame_skip' allows the environment to repeat actions, ignoring the state and summing the reward.
* 'sticky_actions' gives a probability to repeat actions. I'll use this one over frame_skip for now.
* 'pettingzoo_env_to_vec_env' makes a vector environment where there is one vector representing each agent. Since we have 162 agents `(45 ranged + 36 melee) * 2 teams` we get 162 vectors per environment.
* 'concat_vec_envs' concatenates all of the vector environments which will be passed through the model.
* SB3's 'VecMonitor' saves the reward, length, time, etc. to a log file
* SB3's 'VecVideoRecorder' wrapper saves a video whenever the *record_video_trigger* condition is satisfied.
<br /><br />

## Learning Rate Scheduler
This is a learning rate scheduler that I found a good number of people using in Kaggle competitions.

```python
def lr_scheduler(min_lr, max_lr, sw_perc):
    """
    :param min_lr: Minimum learning rate.
    :param max_lr: Maximum learning rate
    :param sw_perc: Progress interval for piecewise function
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining):
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
Setting the minimum learning rate to 1e-8, the maximum to 1e-4, and the interval to 0.2 we get a learning rate schedule that looks like this.

<div class="center">
  <img src="https://filipinogambino.github.io/ngorichs/assets/images/lr_schedule_plot.jpg">
</div>

## Now we can build our model
PPO is an off-policy algorithm so here it consists of 2 seperate models, one for the policy network which yields an agent's action given an observation and another for the value network which yields the expected reward given that same observation. The default networks for SB3's PPO are fully connected layers. Further, the melee agents had their action spaces padded to match that of the ranged agents so we do not need a seperate policy-value pair to accomodate that. That might be something to address in a later post by using embeddings or a transformer to differentiate the types of agents.

<div class="row">
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_policy_net.jpg" alt="Policy Network">
  </div>
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_value_net.jpg" alt="Value Network">
  </div>
</div>
<br />

Now we write a quick config dictionary since we'll need to pass some of the parameters when we initialize WandB and then we can assemble our model.

```python
config = {
    "learning_rate": lr_scheduler(1e-8, 1e-4, 0.2),
    "total_timesteps": int(2e7),
    "log": "runs/baseline",
}

env = make_env()

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

model.learn(total_timesteps=config["total_timesteps"])
wandb.finish()
```
## Results
Let's take a look at a few of the videos at some of the interesting points like steps 0, 12, 24, and 30.
<div class="row">
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/W&B Chart baseline episode mean len.png" alt="Episode mean length">
  </div>
  <div class="column">
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/W&B Chart baseline episode mean rew.png" alt="Episode mean reward">
  </div>
</div>
<br />

First step 0, just to take a look at the initial parameters. Interestingly, the melee units immediately head towards the top of the environment. I would assume this has something to do with the action space padding.
<div class="center">
  <p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_step_0.gif">
  </p>
</div>

Step 12 where the episode length is really decreasing meaning one team is getting eliminated faster. Now the ranged units have also started clumping on one wall. There also seems to be a lot of friendly fire.
<div class="center">
  <p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_step_12.gif">
  </p>
</div>

Step 24 where reward is still increasing, but the episode length has a spike. Still some friendly fire and some agents are attacking out of bounds. There are also quite a few agents just wandering around or stuck behind their allies.
<div class="center">
  <p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_step_24.gif">
  </p>
</div>

Step 30 where the reward is highest. There do seem to be less agents stuck between the boundary and their allies which is great, but their movements towards the enemy could certainly be faster and more efficient. I think the main thing I want to address in the next post is dealing with the action space padding since I believe that is where the clumping on a wall behavior arose when clumping in the middle would be more beneficial so agents don't get stuck.

<div class="center">
  <p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/baseline_step_30.gif">
  </p>
</div>

That's the end of the post, if you want to try it out yourself here is all of the code that you'll need. Just make sure to change the wandb API key.

{% include codeHeader.html %}
```python
# Creating the environment
from pettingzoo.magent import combined_arms_v6

# Wrapping the environment
import supersuit as ss
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor

# For the policy algorithm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

# For logging on Weights and Biases
import wandb
from wandb.integration.sb3 import WandbCallback

# For WandB API key
import os

os.environ["WANDB_API_KEY"]='YOUR_API_KEY_HERE'

def make_env():
    """
    :return: a wrapped, combined arms environment ready for logging on WandB
    """
    env = combined_arms_v6.parallel_env(max_cycles=2000, minimap_mode=True)
    env = ss.pad_action_space_v0(env)
    env = ss.black_death_v3(env)
    env = ss.agent_indicator_v0(env, type_only=True)
    # env = ss.frame_skip_v0(env, (1,3))
    env = ss.sticky_actions_v0(env, 0.3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=2, base_class="gym")
    env = VecMonitor(env, filename=None)
    env = VecVideoRecorder(
        env,
        "videos/",
        record_video_trigger=lambda x: x % 2048 == 0,
        video_length=500,
    )
    return env

def lr_scheduler(min_lr, max_lr, sw_perc):
    """
    :param min_lr: Minimum learning rate.
    :param max_lr: Maximum learning rate
    :param sw_perc: Progress interval for piecewise function
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining):
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

config = {
    "learning_rate": lr_scheduler(1e-8, 1e-4, 0.2),
    "total_timesteps": int(2e7),
    "log": "runs/baseline",
}

env = make_env()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=config["learning_rate"],
    tensorboard_log=config["log"],
    seed=42,
)

wandb.init(
    config=config,
    name="baseline",
    project="combined_arms_v6",
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
)

model.learn(total_timesteps=config["total_timesteps"])
wandb.finish()
```
