---
title: "Convolutional Neural Network"
permalink: /:categories/archive/cnn/
author_profile: false
sidebar:
  nav: "combined_arms"
categories:
  - combined_arms
tags:
  - PettingZoo
  - RL
  - Stable Baselines3
  - PPO
  - WandB
---

In this post we'll be trying out a different architecture. We'll use CNN filters to understand spatial features, some embeddings to differentiate the 4 types of agents, and an LSTM to give our agents an understanding of odometry.

<p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/cnn.jpg">
</p>

To replace the default architecture in stable-baseline3 we need to create a dictionary that will go into the `policy_kwargs` parameter
```python
policy_kwargs = dict(
    features_extractor_class=CombinedArmsFeatures,
    net_arch=[256, 128, 64],
    activation_fn = nn.ReLU,)

env = make_env()
                     
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=lr_scheduler(1e-8, 1e-4, 0.2),
    verbose=1,
    tensorboard_log=config['log'],
    policy_kwargs=policy_kwargs,
)
```

<iframe src="https://wandb.ai/filipinogambino/Combined_Arms_v6/reports/Combined-Arms-Report--VmlldzoyMTI5OTk3?accessToken=bjajeycpq7husvl3jhozn7yo20qo54aw5tut5epw7e0d6uomje62tpbu4ctbufrj" title="WandB" style="border:none; height:512px; width:100%">
</iframe>

