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

In this post we'll be trying out a different architecture. We'll use CNN filters to understand spatial features, embeddings to differentiate the 4 types of agents, and an LSTM to give our agents some understanding of odometry.

<p>
    <img src="https://filipinogambino.github.io/ngorichs/assets/images/drawio/cnn_emb_lstm.jpg">
</p>


```python
class CombinedArmsFeatures(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
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
            obs = obs.transpose([2, 0, 1]) # (Width, Height, Channels) to (Channels, Width, Height)
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
        obs = observations.permute(0,3,1,2)
        
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
```


To replace the default architecture in stable-baseline3 we need to create a dictionary that will go into the `policy_kwargs` parameter
```python
policy_kwargs = dict(
    features_extractor_class=CombinedArmsFeatures,
    net_arch=[128, 128, 64],
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

