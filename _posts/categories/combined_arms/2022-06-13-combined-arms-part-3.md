---
title: "Nevermind, I think I got itt"
categories:
  - combined_arms
tags:
  - PettingZoo
  - RL
  - Stable Baselines3
  - PPO
  - WandB
---


# Convolutional Neural Network
Another architecture I would like to try is a pretty standard CNN with max pooling, batch normalization, and dropout, but I also want to learn the differences between the four types of units so I'll add four of embeddings for that.
<p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/cnn.jpg">
</p>

To replace the default architecture in stable-baseline3 we need to create a dictionary that will go into the `policy_kwargs` parameter
```python
policy_kwargs = dict(
    features_extractor_class=CombinedArmsFeatures,
    net_arch=[256, dict(vf=[256, 128, 64],
                        pi=[128, 64])],
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
<p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/cnn_emb_wandb.jpg">
</p>

<iframe src="https://wandb.ai/filipinogambino/Combined_Arms_v6/reports/Shared-panel-22-06-07-13-06-30--VmlldzoyMTI5OTk3" title="WandB" style="border:none;height:512px;width:100%"></iframe>

