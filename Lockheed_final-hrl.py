'''
Author: Jack McWeeney
Image-augmented maze navigation
Lockheed Martin Collaboration
'''

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights, feature_extraction
from torchvision.io import read_image
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce 
import operator
import random
import itertools
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class fc(nn.Module):
    def __init__(self, features, actions):
        super(fc, self).__init__()
        self.features = features
        self.actions = actions
        self.fc1 = nn.Linear(self.features,700)
        self.fc2 = nn.Linear(700,50)
        self.fc3 = nn.Linear(50, self.actions)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class ReplayMemory(object):
    def __init__(self, model, num_actions=4, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = num_actions

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model(envstate)#torch.FloatTensor(envstate).to(device))

    def get_data(self, env_size, data_size=10):
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = torch.zeros((data_size, env_size)).to(device)
        targets = torch.zeros((data_size, self.num_actions)).to(device)
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            targets[i] = self.predict(envstate).detach()
            Q_sa = torch.max(self.predict(envstate_next)).to(device)
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

class Manager():
    def __init__(self, dims, feature_space, action_space, worker_state, start=(0,0)):
        self.dims = dims
        self.x_dim, self.y_dim = self.dims 
        self.start = start
        self.goal = [self.x_dim-1, self.y_dim-1]
        self.action_space = action_space
        self.feature_space = feature_space
        self.worker_state = worker_state

        self.epsilon = 1
        self.eps_final = 0.01
        self.eps_decay = 0.01
        self.learning_rate = 0.01
        self.reset()
        self.envstate_next = self.envstate()
        self.man_episode = []

        self.net = fc(self.feature_space, len(self.action_space)).to(device)
        self.loss = nn.MSELoss()
        self.optimizer =  torch.optim.SGD(self.net.parameters(), lr = self.learning_rate)
        self.experience = ReplayMemory(self.net, num_actions=len(action_space))

        self.move_dict = {
            0 : (-1,0),
            1 : (0,-1),
            2 : (1,0),
            3 : (0,1),
            4 : (0,0)
        }

    def reset(self):
        self.agent_pos = self.start
        self.tot_reward = 0
        self.n_episodes = 0

    def run(self):
        self.act()
        self.learn()

    #envstate: current manager grid + current worker grid
    def envstate(self):
        env = torch.zeros(self.dims).to(device)
        env[self.agent_pos] = 1
        envstate = torch.cat((env.flatten(), self.worker_state)) #already flattened
        return envstate

    def act(self):
        self.n_episodes+=1
        prev_envstate = self.envstate_next
        if np.random.rand() < self.epsilon:
            action = torch.tensor(random.randrange(len(self.action_space))).to(device)
        else:
            action = torch.argmax(self.experience.predict(prev_envstate)).to(device)
        #L U R D
        new_agent_pos = [sum(tup) for tup in zip(self.agent_pos, self.move_dict[action.item()])]
        oob = (new_agent_pos[0] > self.x_dim-1 or new_agent_pos[1] > self.y_dim-1)
        if (True in (e < 0 for e in new_agent_pos) or oob):
            reward = -0.75
        elif ((self.agent_pos == self.goal) & (new_agent_pos != self.goal)):
            reward = -0.5
        else:
            self.agent_pos = new_agent_pos
            reward = -0.15
        if self.agent_pos == self.goal:
            reward = 1

        self.envstate_next = self.envstate()

        self.man_episode = [prev_envstate, action, reward, self.envstate_next, 0]
        self.experience.remember(self.man_episode)

        self.epsilon = self.eps_final + (1 - self.eps_final) * np.exp(-self.eps_decay*epoch)

        self.learn()

        return self.agent_pos

    def learn(self):
        inputs, targets = self.experience.get_data(env_size = self.feature_space)
        output = self.net(inputs)
        self.optimizer.zero_grad()
        l = self.loss(output, targets)
        l.backward()
        self.optimizer.step()
           
def prod(tup):
        return reduce(operator.mul, tup, 1)

class Maze():
    def __init__(self, dims, img_dir, action_space, feature_space, return_node, start=(0,0), hrl=False):
        self.dims = dims
        self.x_dim, self.y_dim = self.dims 
        self.start = start
        self.agent_pos = self.start
        self.goal = [self.x_dim-1, self.y_dim-1]
        self.img_dir = img_dir
        self.action_space = action_space
        self.feature_space = feature_space
        self.return_node = return_node
        self.epsilon = 1
        self.eps_final = 0.01
        self.eps_decay = 0.01
        self.learning_rate = 0.01
        self.min_reward = -0.5 * np.prod(dims)**2
        self.hrl = hrl
        if self.hrl:
            self.manager = Manager([2,2], (prod(self.dims)+(2*2)), [range(5)], self.get_state())
            self.manager_pos = (0,0)
            self.manager_state = torch.zeros(self.manager.dims).to(device)
            self.manager_state[self.manager_pos] = 1
            self.feature_space += prod(self.manager.dims)

        self.reset()

        self.run_eps, self.man_eps, self.run_rewards, self.run_wins = [], [], [], []

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.preprocess = weights.transforms()

        model = resnet50(weights=weights).to(device)
        model.eval()
        self.feat_ext = feature_extraction.create_feature_extractor(model, return_nodes=[self.return_node]).to(device)
        self.train_net = fc(self.feature_space, len(self.action_space)).to(device)
        self.loss = nn.MSELoss()
        self.optimizer =  torch.optim.SGD(self.train_net.parameters(), lr = self.learning_rate)
        self.experience = ReplayMemory(self.train_net, num_actions=len(action_space))
        self.image_feats = [torch.zeros(1).to(device)]*(prod(dims))
        self.compute_envstates()

        self.envstate_next = self.ind_envstate()

        self.img_transform = torch.nn.Sequential(
            T.RandomPerspective(distortion_scale=0.6, p=0.2),
            T.ColorJitter(brightness=.5, contrast=.3,  hue=.3),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))            
        )
   
        self.move_dict = {
            0 : (-1,0),
            1 : (0,-1),
            2 : (1,0),
            3 : (0,1),
        }

    def reset(self):
        self.agent_pos = self.start
        self.tot_reward = 0
        self.game_over, self.win = False, False
        self.n_episodes = 0
        if self.hrl: self.manager.reset()

    def run(self, transform=False):
        self.reset()
        while not self.game_over:
            if self.hrl: 
                self.manager.worker_state = self.get_state()
                self.manager_pos = self.manager.act()
                self.manager.learn()
            for _ in range(10):
                self.act(transform)
                self.learn()
                if self.game_over: break
        
        print(f'Epoch: {epoch} | Win: {self.win} | Episodes: {self.n_episodes} | Reward: {self.tot_reward}')
        self.run_wins.append(self.win)
        self.run_rewards.append(self.tot_reward)
        self.run_eps.append(self.n_episodes)
        self.man_eps.append(self.manager.n_episodes)
        win_len = max(-10, -len(self.run_wins))
        writer.add_scalar("Worker/win rate over 10 eps (100x scale)", (sum(self.run_wins[win_len:])/abs(win_len))*100, len(self.run_wins))
        writer.add_scalar("Worker/Total rewards per episode", self.run_rewards[-1], len(self.run_rewards))
        writer.add_scalar("Worker/Run length", self.run_eps[-1], len(self.run_eps))
        writer.add_scalar("Epsilon", self.epsilon, len(self.run_eps))
        
    def img_ind(self):
        return ((self.y_dim) * (self.agent_pos[0]) + self.agent_pos[1])

    def compute_envstates(self):
        for i in range(np.prod(self.dims)):
            #img_ind = self.img_ind()
            img_file = os.path.join(f'{self.img_dir}{i+1}.jpg')
            try: img = read_image(img_file)
            except: breakpoint()
            img_ext = self.preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = self.feat_ext(img_ext)
            self.image_feats[i] = features[self.return_node].flatten()
            if self.hrl:
                self.image_feats[i] = torch.cat((self.image_feats[i], self.get_state(man=True)))
            return features[self.return_node].flatten()

    def ind_envstate(self):
        return self.image_feats[self.img_ind()]

    def trans_envstate(self):
        img_ind = self.img_ind()
        img_file = os.path.join(f'{self.img_dir}{img_ind+1}.jpg')
        try: img = read_image(img_file)
        except: breakpoint()
        img = self.img_transform(img)
        img_ext = self.preprocess(img).unsqueeze(0).to(device)
        #breakpoint()
        with torch.no_grad():
            features = self.feat_ext(img_ext)

        if self.hrl:
            return torch.cat((features[self.return_node].flatten(), self.get_state(man=True))) #already flattened
        
        return features[self.return_node].flatten()
        
    def act(self, transform = False):
        self.n_episodes+=1
        if self.n_episodes%100 == 0: print(self.n_episodes)
        prev_envstate = self.envstate_next
        if np.random.rand() < self.epsilon:
            action = torch.tensor(random.randrange(len(self.action_space))).to(device)
        else:
            action = torch.argmax(self.experience.predict(prev_envstate)).to(device)
        #L U R D
        new_agent_pos = [sum(tup) for tup in zip(self.agent_pos, self.move_dict[action.item()])]
        oob = (new_agent_pos[0] > self.x_dim-1 or new_agent_pos[1] > self.y_dim-1)
        if (True in (e < 0 for e in new_agent_pos) or oob):
            reward = -0.75
        else:
            self.agent_pos = new_agent_pos
            reward = -0.15
        
        ratio = (self.x_dim/self.manager.x_dim, self.y_dim/self.manager.y_dim)
        quadrant = [int(x/y) for x, y in zip(self.agent_pos, ratio)]

        if quadrant == self.manager.agent_pos: 
            reward = reward/2
        
        if self.agent_pos == self.goal:
            self.win = True
            self.game_over = True
            reward = 1
            if self.manager.agent_pos == quadrant:
                man_reward = 1
            else:
                man_reward = -1
            man_ep = self.manager.man_episode
            man_ep[-1] = 1
            man_ep[2] = man_reward
            self.manager.experience.remember(man_ep)

        if self.tot_reward < self.min_reward:
            self.game_over = True
        self.tot_reward += reward
        #experience - [envstate, action, reward, envstate_next, game_over]
        if transform:
            self.envstate_next = self.trans_envstate()
        else:
            self.envstate_next = self.ind_envstate()
        if len(self.envstate_next) > self.feature_space: breakpoint()
        episode = [prev_envstate, action, reward, self.envstate_next, self.game_over]
        self.experience.remember(episode)

        self.epsilon = self.eps_final + (1 - self.eps_final) * np.exp(-self.eps_decay*epoch)

    def learn(self):
        try: inputs, targets = self.experience.get_data(env_size = self.feature_space)
        except: breakpoint()
        output = self.train_net(inputs)
        self.optimizer.zero_grad()
        l = self.loss(output, targets)
        l.backward()
        self.optimizer.step()

    def get_state(self, man=False):
        if man:
            dim = self.manager.dims
            pos = self.manager_pos
        else: 
            dim = self.dims
            pos = self.agent_pos
        env = torch.zeros(dim).to(device)
        env[pos] = 1
        return env.flatten()

def show(dims, agent_pos):
    plt.figure(1)
    plt.grid('on')
    plt.clf()
    nrows, ncols = dims
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.ones([nrows, ncols])
    agent_col = agent_pos[0]
    agent_row = agent_pos[1]
    canvas[agent_row, agent_col] = 0.5   # agent cell
    canvas[nrows-1, ncols-1] = 0.9 # goal cell
    plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.grid()
    plt.plot()
    plt.pause(0.001)

if __name__ == "__main__":
    manager_maze_size = [2,2]
    worker_maze_size = [8,8] #up to 8x8/total 64 images
    feature_spaces = {
        # 'layer1' : 3136, #56**2 ###torch.Size([1, 256, 56, 56]) > 802,816
        # 'layer2' : 784, #28**2
        # 'layer3' : 196, #14**2
        'layer4' : 100352 #7**2 ### torch.Size([1, 2048, 7, 7]) > 100,352
    }
    action_space = [0,1,2,3]
    return_nodes = ['layer4'] #['layer1', 'layer2', 'layer3', 'layer4']
    img_dir = './streetview/'
    writer = SummaryWriter(comment='no_pos-decay0.01')

    for i in range(len(return_nodes)):
        feature_space = feature_spaces[return_nodes[i]]#+2
        n_epoch = 1000
        worker = Maze(worker_maze_size, img_dir, action_space, feature_space, return_nodes[i], hrl=True)
        print(f'Total epochs = {n_epoch}')
        for epoch in range(n_epoch):
            worker.run(transform = True)

    writer.flush()
    writer.close()