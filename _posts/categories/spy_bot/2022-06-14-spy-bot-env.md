---
title: "Building the Environment"
permalink: /:categories/archive/env/
author_profile: false
sidebar:
  nav: "spy_bot"
categories:
  - spy_bot
tags:
  - Gym
  - RL
  - SB3
  - PPO
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

This page is under development


```python
class MarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nfeats):
        super(MarketEnv, self).__init__()
        self.n_actions = 3 # {Buy: 0, Sell: 1, Hold: 2}
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(nfeats,CONFIG.BS))

        self.cash = CONFIG.CASH
        self.shares_owned = 0
        self.reward = 0
        self.done = 0
        self.n_trades = 0
        self.close = None
        self.prev_close = 0.672762 # First close value (normalized)
        self.risum = 0
        self.rfsum = 0
        self.initial_bh = self.prev_close # initial buy and hold price
        
    def step(self, actions, states):
        num_shares = 10
        buy = False
        sel = False
        reward = [None] * actions.size(0)
        done = [None] * actions.size(0)
        trades = [None] * actions.size(0)
        closes = states[:,19]
        self.acc_values = [None] * actions.size(0)
        
        for idx,element in enumerate(zip(actions, closes)):
            action, self.close = element

            if action==0 and not buy: # Buy
                buy = True
                sel = False
                self.n_trades += 1
                self.cash -= self.close * num_shares # Cost basis
                self.shares_owned += num_shares
            elif action==1 and not sel: # Sell
                buy = False
                sel = True
                self.n_trades += 1
                self.cash += num_shares * self.close
                self.shares_owned -= num_shares
            elif action==2:
                buy = False
                sel = False
            
#             self.reward = self.reward_math(buy, sel)
            if idx==0:
                self.reward = 0
            else:
                if self.acc_values[idx-1] < (self.cash + self.shares_owned * self.close):
                    self.reward = -1
                elif self.acc_values[idx-1] > (self.cash + self.shares_owned * self.close):
                    self.reward = 1
                else:
                    self.reward = 0
            self.done = 1 if self.cash < 0 else 0 # Need enough cash to open a position
            self.acc_values[idx] = self.cash + self.shares_owned * self.close
            
            reward[idx] = self.reward
            done[idx] = self.done
            self.prev_close = self.close
        
        rewards = torch.tensor(reward)
        dones = torch.tensor(done)
        
        return [rewards, dones]

#     https://ai.stackexchange.com/questions/10082/suitable-reward-function-for-trading-buy-and-sell-orders/10912
    def reward_math(self, buy, sel):
        fees = 0.0025 # Fees associated with making a trade. Set to 0.25% per trade
        excess_trading_loss = math.log((1-fees)/(1+fees))
        log_close = math.log(self.close)
        log_prev_close = math.log(self.prev_close)
        self.risum += buy * (log_close - log_prev_close)
        self.rfsum += sel * 0.1 * self.prev_close / 525_600 # 10% * previous minute's closing price / minutes per year (i.e. Baseline growth per share)
        r = self.risum + self.rfsum
        rbh = self.close - self.initial_bh + excess_trading_loss
        return math.tanh(0.01*(r - rbh))

    def reset(self):
        self.cash = CONFIG.CASH
        self.acc_value = CONFIG.CASH
        self.n_trades = 0
        self.shares_owned = 0
        self.done = 0
        self.risum = 0
        self.rfsum = 0
        
        return torch.zeros(CONFIG.BS,nfeats)
```
