---
title: "Baseline"
permalink: /:categories/archive/baseline/
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
class DeepQNetwork(nn.Module):
    def __init__(self, model, n_actions, n_cont, emb_dims, hidden_size, nlayers, name, chkpt_dir):
        # https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
        super(DeepQNetwork, self).__init__()
        self.model = model
        self.emb_layers = nn.ModuleList([nn.Embedding(i,j) for i,j in emb_dims])
        
        self.n_embs = sum([j for _,j in emb_dims])
        self.n_cont = n_cont
        n_feats = self.n_cont + self.n_embs
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        if model=='lstm':
            self.lstm = nn.LSTM(n_feats, hidden_size, self.nlayers, batch_first=True)
            self.hn = torch.zeros((self.nlayers,CONFIG.SEQ_LEN,self.hidden_size), device=CONFIG.DEVICE)
            self.cn = torch.zeros((self.nlayers,CONFIG.SEQ_LEN,self.hidden_size), device=CONFIG.DEVICE)
        elif model=='gru':
            self.gru = nn.GRU(n_feats, hidden_size, self.nlayers, batch_first=True)
            self.hn = torch.zeros((self.nlayers,CONFIG.SEQ_LEN,self.hidden_size), device=CONFIG.DEVICE)
            
        self.fc_1 = nn.Linear(hidden_size, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, n_actions)
        self.relu = nn.ReLU()
        
        self.optimizer = optim.Adam(self.parameters(), lr=CONFIG.LR)
        
        self.to(CONFIG.DEVICE)
            
    def forward(self, state):
        bs = state.size(0)
        cat_data = state[:,:15].long()
        cont_data = state[:,15:]

        emb = [emb_layer(cat_data[:,i]) for i,emb_layer in enumerate(self.emb_layers)]
        emb_ = torch.cat(emb,1)
        
        input_ = torch.cat([emb_, cont_data],1).float()
        input_ = input_.unsqueeze(0)
        if self.model=='lstm':
            output, (self.hn, self.cn) = self.lstm(input_, (self.hn.detach(),self.cn.detach())) # self.hn, self.cn
        elif self.model=='gru':
            output, self.hn = self.gru(input_, self.hn.detach()) # self.hn
        output = output.contiguous().view(bs,-1)
        out = self.relu(output)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        return torch.softmax(out,1)
```
