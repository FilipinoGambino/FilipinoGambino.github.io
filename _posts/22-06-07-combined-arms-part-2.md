---
title: "Combined Arms Part 2"
categories:
  - Blog
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


# Convolutional Neural Network
Another architecture I would like to try is a pretty standard CNN with max pooling, batch normalization, and dropout, but I also want to learn the differences between the four types of units so I'll add four of embeddings for that.
<p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/cnn.jpg" width="350" height="350">
</p>

To replace the default architecture in stable-baseline3 we need to create a dictionary that will go into the `policy_kwargs` parameter
```python
policy_kwargs = dict(features_extractor_class=CombinedArmsFeatures,
                     net_arch=[256, dict(vf=[256, 128, 64],
                                         pi=[128, 64])],
                     activation_fn = nn.ReLU,)
```
