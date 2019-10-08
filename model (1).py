#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnetwork(nn.Module):
    
    def __init__(self,state_size,action_size,seed,fc_1=64,fc_2=64):
        
        super(Qnetwork,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,fc_1)
        self.fc2 = nn.Linear(fc_1,fc_2)
        self.fc3 = nn.Linear(fc_2,action_size)
    
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# In[ ]:




