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

In this post we'll be trying out a different policy architecture that should be able to gather more information about the environment. We'll use CNN filters to understand spatial features, embeddings to differentiate the 4 types of agents, and an LSTM to give our agents some understanding of odometry.

<p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/cnn_emb_lstm.jpg">
</p>

This architecture was heavily influenced by [Samuel Shaw et al.'s paper "ForMIC: Foraging via Multiagent RL with Implicit Communication"](https://arxiv.org/pdf/2006.08152.pdf) and my next post will likely be integrating some form of pheromones into this policy.


# Environment Wrapper Changes
The two wrappers we're going to add are *observation_lambda_v0* and *frame_stack_v1*. The first wrapper is supersuit's own observation lambda function that we can use to write our own wrapper and we'll use it to transpose the observation to put channels first (<code>from (Height, Width, Channels) to (Channels, Height, Width)</code>) since that is what the pytorch Conv2D layers are expecting. The latter wrapper simply stacks a set of observation frames (at timesteps t, t-1,...,t-n where n=stack_size-1) to be used in the LSTM cell. We can also remove the sticky_actions wrapper from the part 1 post because our agents have memory now. I'd be interested to see if sticky_actions actually helped the agents eliminate enemy agents in the early training stages.

```python
def make_env():
    env = combined_arms_v6.parallel_env(
        max_cycles=1000,
        minimap_mode=True,
    )
    env = ss.frame_stack_v1(env, stack_size=2)
    env = ss.pad_action_space_v0(env)
    env = ss.black_death_v3(env)
    env = ss.agent_indicator_v0(env, type_only=True)
    env = ss.observation_lambda_v0(env, lambda obs, obs_space : obs.transpose((2,0,1)))
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, config.num_envs, num_cpus=config.num_envs // 2, base_class="gym")
    env = VecMonitor(env, filename=None)
    env = VecVideoRecorder(env, config.video_logs, record_video_trigger=lambda x: x % 2048 == 0, video_length=500)
    return env
```

# Building the Policy
The policy will be assembled in the following class and inherit from pytorch's nn.Module class.
```python
class CombinedArmsFeatures(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CombinedArmsFeatures, self).__init__()
        
    def forward(self, observations: torch.Tensor):
```

Starting with the convolution filters, we want to increase the number of filters to extract information while allowing our max pooling layers to decrease the height and width of the input frames. Of course, we're also constrained by training speed so we'll keep our filter count resonable. We'll be putting both the time *t* and *t-1* agent observations through the CNN, so we also use a flatten layer at the end so we can concatenate the observations for the LSTM layer.

```python
self.cnn = nn.Sequential(
    nn.Conv2d(inp_channels, 28, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(28, 28, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(3, stride=2),
    nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(56, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.Flatten(),
)
```

Further, we need to know the shape of the flatten layer's output so that the next layer knows what input size it should expect. I'll note that technically we don't need the flatten layer here since this architecture's final convolution layer outputs a 64x1x1, but if we want to change anything in the cnn, we'll likely need it. Anyway, we can get output shape dynamically by grabbing a sample observation and running it through the CNN. For that, a batch dimension is necessary so we add one with <code>obs[None]</code> and convert it to a pytorch tensor.

```python
with torch.no_grad():
    obs = observation_space.sample()
    obs = obs[:15,:,:] # Drop the encoding channels and time t observation channels
    n_flatten = self.cnn(torch.as_tensor(obs[None]).float()).shape[1]
```

Moving on to the LSTM; this one is pretty simple. 64 features in and 64 features out. We'll just let pytorch handle the cell and hidden states.

```python
self.lstm = nn.LSTM(64, 64, batch_first=True)
```

For the embeddings, the input starts as a 4x13x13 one-hot encoding for each of the four agent types, so we remove all of the excess/redundant information and pass along a 4x1 one-hot encoding.

```python
n_agent_types = 4
emb_dim = 8
self.emb = nn.Sequential(
    nn.Embedding(n_agent_types, emb_dim),
    nn.ReLU(),
    nn.Linear(emb_dim, n_flatten),
    nn.ReLU(),
)
```

Tieing all of these components together, we concatenate them and send them through a series of fully connected layers. There are only two such layers here, but SB3 has another way of assembling the seperate action and value policies which plays into the next step.

```python
self.fc = nn.Sequential(
    nn.Linear(n_flatten * (n_agent_types + 2), 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
)
```

In SB3, we can use our own new "policy feature extractor" by passing it in a dictionary. We can also pass in the dimensions of the linear layers for the action policy and value policy in the *net_arch* variable along with the activation function in *activation_fn*.

```python
policy_kwargs = dict(
    features_extractor_class=CombinedArmsFeatures,
    net_arch=[256, 128, 64],
    # net_arch=[128, dict(vf=[256], pi=[16])],
    activation_fn = nn.ReLU,
    )
```

One last thing to change from the last post is replacing "MlpPolicy" with "CnnPolicy" so that we receive the observations with the correct shape. Now, all that's left to do is call the SB3's PPO class and train our policy.
```python
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

# Results
<iframe src="https://wandb.ai/filipinogambino/Combined_Arms_v6/reports/Episode-Reward-Mean-22-06-23-15-06-43---VmlldzoyMjE2OTY3" style="border:none;height:1024px;width:100%">
</iframe>
Unfortunately, it looks like our policy was not able to outperform the baseline policy. What's more is that each run at 20,000,000 steps with 4 parallel environments takes about 1.3 days to train and thus has gotten rather expensive. I would love to manually play around with the parameters of this model or try out some hyperparameter tuning, but my wallet says I need to just move on to the pheromones paper. 🙃


Here's everything put together.

```python
class CombinedArmsFeatures(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CombinedArmsFeatures, self).__init__()

        emb_dim = 8
        n_agent_types = 4 # Range and melee for both teams
        inp_channels = (observation_space.shape[-1] - n_agent_types) // 2
        self.features_dim = features_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(inp_channels, 28, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(28, 28, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(56, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Flatten(),
        )
        
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        
        with torch.no_grad():
            obs = observation_space.sample()
            obs = obs[:15,:,:] # Drop the encoding channels
            n_flatten = self.cnn(torch.as_tensor(obs[None]).float()).shape[1]
        
        self.emb = nn.Sequential(
            nn.Embedding(n_agent_types, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, n_flatten),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(n_flatten * (n_agent_types + 2), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.device = config.device
        
    def forward(self, observations: torch.Tensor):
        obs_t0 = obs[:,:15,:,:]
        obs_t1 = obs[:,15:-4,:,:]
        obs_emb = obs[:,-4:,0,0].long()
        
        cnn_t0 = self.cnn(obs_t0).unsqueeze(1)
        cnn_t1 = self.cnn(obs_t1).unsqueeze(1)
        cnn = torch.cat([cnn_t0, cnn_t1], axis=1)

        mem,_ = self.lstm(cnn)
        emb = self.emb(obs_emb)
        
        out = torch.cat([cnn, emb], axis=1)
        out = out.view(obs.shape[0], -1)
        
        return self.fc(out)

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
