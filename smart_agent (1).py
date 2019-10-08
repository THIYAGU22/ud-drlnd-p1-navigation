


import numpy as np
import random
from collections import deque
from collections import namedtuple,deque

from model import Qnetwork

import torch 
import torch.nn.functional as F
import torch.optim as optim


## GPU on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount RATE
LR = 5e-4               # lr
TAU = 1e-3              # tau for update in target parameters
UPDATE_EVERY = 4        # for every 4 steps update





class smart_agent():
    
    def __init__(self,state_size,action_size,seed):
        
        self.state_size = state_size
        self.action_size=action_size
        #self.device=device
        self.seed = random.seed(seed)
        
        
        self.q_network_local = Qnetwork(state_size,action_size,seed).to(device)
        self.q_network_target = Qnetwork(state_size,action_size,seed).to(device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(),lr=LR)
        
        ####REPLAY MEMORY########
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,seed)
        
        self.t_step = 0
        
    def step(self,state,action,reward,next_state,done):
        self.memory.add(state,action,reward,next_state,done)
        self.t_step = (self.t_step + 1)%UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                xp = self.memory.sample()
                self.learn(xp,GAMMA)
    
    def act(self,state,eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network_local.eval()
        
        with torch.no_grad():
            action_value=self.q_network_local(state)
        self.q_network_local.train()
        
        #Epsilon greedy selection
        if random.random()>eps:
            return np.argmax(action_value.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self,xp,gamma):
        state,action,reward,next_state,done = xp
        q_target_next = self.q_network_target(next_state).detach().max(1)[0].unsqueeze(1)
        q_target = reward + ( gamma * q_target_next * (1 - done))
        
        q_expected = self.q_network_local(state).gather(1,action)
        
        #MSE LOSS
        loss = F.mse_loss(q_expected,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.q_network_local,self.q_network_target,TAU)
    
    def soft_update(self,local_model,target_model,tau):
        for target_param,local_param in zip(target_model.parameters(),
                                            local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 - tau) * target_param.data)
            
    
       
class ReplayBuffer():
    def __init__(self,action_size,buffer_size,batch_size,seed):
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.xp = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.seed = random.seed(seed)
        
        
    def add(self,state,action,reward,next_state,done):
        e = self.xp(state,action,reward,next_state,done)
        self.memory.append(e)
        
    
    def sample(self):
        xps = random.sample(self.memory,k = self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in xps if e is not None])).float().to(device)
        action = torch.from_numpy(np.vstack([e.action for e in xps if e is not None])).long().to(device)
        reward = torch.from_numpy(np.vstack([e.reward for e in xps if e is not None])).float().to(device)
        next_state = torch.from_numpy(np.vstack([e.next_state for e in xps if e is not None])).float().to(device)
        done = torch.from_numpy(np.vstack([e.done for e in xps if e is not None]).astype(np.uint8)).float().to(device)
        
        return states,action,reward,next_state,done

    def __len__(self):
        return len(self.memory)
    


    

