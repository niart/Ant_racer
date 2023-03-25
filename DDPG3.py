## this file is used to train the 2nd agent (middle, ant) after the 1st agent (big, spider) has NOT been trained
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
from itertools import count
import random
import math
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Exp = namedtuple('Exp', ('state', 'action', 'reward', 'nextState', 'done'))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"


# OU noisy
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, actionSize, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = int(actionSize)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0 #for recursive array

    # the length of buffer
    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Exp(*args)  # add
        self.position = (self.position + 1) % self.capacity  # update index

    # sample from buffer
    def sample(self, batchSize):
        return random.sample(self.buffer, batchSize)
    
    def fill(self, env, initLen, maxSteps):
        stateSize = env.observation_space.shape[0]
        if initLen > self.capacity:
            return
        while len(self.buffer) < initLen:
            state = env.reset()
            for t in count():
                action = env.action_space.sample()
                nextState, reward, done, _ = env.step(action)
                #reward = -reward
                action = action[:8]
                env.render()
                self.push(torch.tensor(state.reshape(1, stateSize), device=device, dtype=torch.float),
                          torch.tensor(action.reshape(1, -1), device=device, dtype=torch.float),
                          torch.tensor(np.array([reward]).reshape(1, -1), device=device, dtype=torch.float),
                          torch.tensor(nextState.reshape(1, stateSize), device=device, dtype=torch.float),
                          torch.tensor(np.array([not done]).reshape(1, -1), device=device, dtype=torch.long))
                state = nextState
                if done or t + 1 >= maxSteps:
                    break


class Actor(nn.Module):
    def __init__(self, stateSize, hiddenSize, actionSize, lim):
        super(Actor, self).__init__()
        self.lim = lim
        self.linear1 = nn.Linear(stateSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, hiddenSize // 2)
        self.linear3 = nn.Linear(hiddenSize // 2, actionSize)
    
    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        action = torch.tanh(self.linear3(x))#.cpu()#.detach().numpy()
        action = action * torch.tensor(self.lim, device=device) ##device=device added by cyy
        return action


class Critic(nn.Module):
    def __init__(self, stateSize, hiddenSize, actionSize):
        super(Critic, self).__init__()
        self.state1 = nn.Linear(stateSize, hiddenSize)
        self.state2 = nn.Linear(hiddenSize, hiddenSize // 2)
        self.action1 = nn.Linear(actionSize, hiddenSize // 2)

        self.linear1 = nn.Linear(hiddenSize, hiddenSize // 2)
        self.linear2 = nn.Linear(hiddenSize // 2, 1)

    def forward(self, state, action):
        s1 = F.leaky_relu(self.state1(state))
        s2 = F.leaky_relu(self.state2(s1))
        a1 = F.leaky_relu(self.action1(action))

        x = torch.cat((s2, a1), dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x


class DDPG:
    def __init__(self, env, buf, actorLR, criticLR, gamma, tau, batchSize, hiddenSize, maxSteps, maxEps, updateStride):
        self.env = env
        self.buffer = buf
        self.actorLR = actorLR
        self.criticLR = criticLR
        self.gamma = gamma
        self.tau = tau
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.maxSteps = maxSteps
        self.maxEps = maxEps
        self.noise = OrnsteinUhlenbeckActionNoise(int(self.env.action_space.shape[0]/3))
        self.updateStride = updateStride

        self.actor = Actor(self.env.observation_space.shape[0], self.hiddenSize, int(self.env.action_space.shape[0]/3), self.env.action_space.high[:8]).to(device)
        self.actorTarget = Actor(self.env.observation_space.shape[0], self.hiddenSize, int(self.env.action_space.shape[0]/3), self.env.action_space.high[:8]).to(device)
        self.critic = Critic(self.env.observation_space.shape[0], self.hiddenSize, int(self.env.action_space.shape[0]/3)).to(device)
        self.criticTarget = Critic(self.env.observation_space.shape[0], self.hiddenSize, int(self.env.action_space.shape[0]/3)).to(device)

        self.actorOpt = optim.Adam(self.actor.parameters(), self.actorLR)
        self.criticOpt = optim.Adam(self.critic.parameters(), self.criticLR)

        self._updateParams(self.actorTarget, self.actor)
        self._updateParams(self.criticTarget, self.critic)
        """
        self.actor_3 = Actor(self.env.observation_space.shape[0], self.hiddenSize,
                           int(self.env.action_space.shape[0] / 3), self.env.action_space.high[-8:]).to(device)
        self.actor_3.load_state_dict(torch.load('push_netDDPG_A.pkl'))
        """
    def _updateParams(self, t, s, isSoft=False):
        for t_param, param in zip(t.parameters(), s.parameters()):
            if isSoft:
                t_param.data.copy_(t_param.data * (1.0 - self.tau) + param.data * self.tau)
            else:
                t_param.data.copy_(param.data)
    
    def _getAction(self, state):
        action = self.actor.forward(torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float)) + torch.tensor(self.noise.sample()* self.env.action_space.high[:8], device=device, dtype=torch.float)
        #return action.detach().cpu().numpy()
        #return action.detach().cpu().numpy().to(device)
        return action.detach().to(device).cpu().numpy() ## same as cyy_vs
    
    def learn(self):
        if len(self.buffer) < self.batchSize:
            print("Can't fetch enough exp!")
            return
        exps = self.buffer.sample(self.batchSize)
        batch = Exp(*zip(*exps))  # batch => Exp of batch
        stateBatch = torch.cat(batch.state) # batchSize * stateSpace.shape[0]
        actionBatch = torch.cat(batch.action) # batchSize * 1
        rewardBatch = torch.cat(batch.reward)  # batchSize * 1
        nextStateBatch = torch.cat(batch.nextState)  # batchSize * stateSpace.shape[0]
        doneBatch = torch.cat(batch.done)

        nextActionBatch = self.actor.forward(nextStateBatch).detach()
        targetQ = self.criticTarget.forward(nextStateBatch, nextActionBatch).detach().view(-1)
        y = rewardBatch.view(-1) + doneBatch.view(-1) * self.gamma * targetQ
        Q = self.critic.forward(stateBatch, actionBatch).view(-1)
        critic_loss = F.smooth_l1_loss(Q, y)

        self.criticOpt.zero_grad()
        critic_loss.backward()
        self.criticOpt.step()

        mu = self.actor.forward(stateBatch)
        actor_loss = -1.0 * torch.sum(self.critic.forward(stateBatch, mu))
        self.actorOpt.zero_grad()
        actor_loss.backward()
        self.actorOpt.step()

        self._updateParams(self.actorTarget, self.actor, isSoft=True)
        self._updateParams(self.criticTarget, self.critic, isSoft=True)

    def train(self):
        res = []
        N = 0
        print("Now the training starts!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for eps in range(self.maxEps):
            state = self.env.reset()
            ret = 0.0
            for t in count():
                action = self._getAction(state).squeeze() # middle agent
                action_all = np.insert(self.env.action_space.sample()[8:], 0, action) #insert small and big agent
                nextState, reward, done, _ = self.env.step(action_all)
                #reward = -reward
                self.env.render()
                self.buffer.push(torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float),
                                 torch.tensor(action.reshape(1, -1), device=device, dtype=torch.float),
                                 torch.tensor(np.array([reward]).reshape(1, -1), device=device, dtype=torch.float),
                                 torch.tensor(nextState.reshape(1, -1), device=device, dtype=torch.float),
                                 torch.tensor(np.array([not done]).reshape(1, -1), device=device, dtype=torch.long))
                if N % self.updateStride and N > 0:
                    self.learn()
                ret += reward
                N += 1
                if done or t + 1 >= self.maxSteps:
                    print("Eps: %d\tRet: %f\tSteps: %d" % (eps + 1, ret, t + 1))
                    self.noise.reset()
                    break
                state = nextState
            res.append(ret)

        np.save('ant_chase_resDDPG', res)
        plt.plot(res)
        d = {'con1': res}
        df = pd.DataFrame(data=d)
        df.to_excel('ant_chase_ddpg.xlsx')

        plt.ylabel('Return')
        plt.xlabel('Episodes')
        plt.savefig('ant_chase_resDDPG.png')
        torch.save(self.actor.state_dict(), 'ant_chase_netDDPG_A.pkl')
        torch.save(self.critic.state_dict(), 'ant_chase_netDDPG_C.pkl')
        self.env.close()

def runDDPG(env_name):
    env = gym.make(env_name)
    buf = ReplayBuffer(1000000)
    buf.fill(env, 10000, 200)  # 1000 means initial length ; 200 max steps #10000
    ddpg = DDPG(env=env,
                buf=buf,
                actorLR=1e-4,
                criticLR=1e-3,
                gamma=0.99,
                tau=0.01,
                batchSize=64,
                hiddenSize=512,
                maxSteps=600,
                maxEps=10000, #maxEps=2000,
                updateStride=30)
    ddpg.train()


