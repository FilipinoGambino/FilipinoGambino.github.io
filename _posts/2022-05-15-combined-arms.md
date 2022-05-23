---
title: "PettingZoo and MAgent's Combined Arms"
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
---

Hello and welcome to my first blog post! Today I'm going to be working with the multi-agent [Combined Arms](https://www.pettingzoo.ml/magent/combined_arms) environment on [PettingZoo](https://www.pettingzoo.ml/#) to build up a basis for implementing papers in the near future.

I'm going to be using a variety wrappers from the supersuit library to improve training speed and performance.

```python
import supersuit as ss
from pettingzoo.magent import combined_arms_v6

env = combined_arms_v6.parallel_env(map_size=45,
                                    max_cycles=1000, minimap_mode=True,
                                    extra_features=False,
                                    step_reward=-0.01,
                                    dead_penalty=-0.2,
                                    attack_penalty=-0.2,
                                    attack_opponent_reward=0.2)
                                    
env = ss.black_death_v3(env)
env = ss.agent_indicator_v0(env, type_only=True)
env = ss.frame_skip_v0(env, (1,3))
env = ss.pad_action_space_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 16, num_cpus=8, base_class='stable_baselines3')
```

1. The *black_death* wrapper allows the environment to remove deceased agents which prevents unecessary calculations.
2. The *agent_indicator* wrapper adds one layer to the observation space for each type of agent. In this case there is a friendly ranged, friendly melee, enemy range, enemy melee.
3. The *frame_skip wrapper* allows the environment to repeat actions, ignoring the state and summing the reward.
4. The *pad_action_space* wrapper makes the each agent's action space the same.
5. The *pettingzoo_env_to_vec_env* wrapper makes a vector environment where there is one vector representing each agent.
6. The *concat_vec_envs* wrapper concatenates all of the vector environments which will be passed to through our model.

```python
policy_kwargs = dict(features_extractor_class=CombinedArmsFeatures,
                     # features_extractor_kwargs=dict(features_dim=256),
                     net_arch=[256, dict(vf=[256, 128, 64],
                                         pi=[128, 64])],
                     activation_fn = nn.ReLU,)

model = PPO("CnnPolicy",
            env,
            verbose=1,
            gamma=0.95,
            n_steps=64,
            ent_coef=9e-2,
            learning_rate=1e-4,
            vf_coef=4e-2,
            max_grad_norm=0.99,
            gae_lambda=0.99,
            n_epochs=8,
            batch_size=256,
            policy_kwargs=policy_kwargs,
            seed=42,)
```



TODO:
Finish this post
