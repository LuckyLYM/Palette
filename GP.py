import os
import sys
import time
import numpy as np
import random
import math
import pickle
import re
import copy
import utils

from sklearn.gaussian_process import GaussianProcessRegressor




class GPUCB(object):

  def __init__(self, grid, environment, beta=50.):

    self.grid = grid
    self.environment = environment
    self.beta = beta

    self.X_grid = self.grid.T
    self.num=self.X_grid.shape[0]
    self.mu = np.array([0. for _ in range(self.num)] )
    self.sigma = np.array([0.5 for _ in range(self.num)])
    self.X = []
    self.T = []
    self.gp=GaussianProcessRegressor()


  def ucb(self):
    return np.argmax(self.mu + self.sigma * np.sqrt(self.beta)), np.max(self.mu + self.sigma * np.sqrt(self.beta))

  def sample(self, x):
    count,reward,cost = self.environment.sample(x)  
    self.X.append(self.X_grid[x])
    self.T.append(reward)
    return count,cost

  def selection(self, K):
    total=0
    samples=0
    for i in range(100):
        if len(self.T)==0:
            grid_idx=random.randint(0,self.num-1)
        else:
            grid_idx, ucb = self.ucb()

        count,cost=self.sample(grid_idx)
        if count==2:
          break

        samples=samples+1

        total=total+cost
        self.gp.fit(self.X, self.T)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)

    index=np.argsort(-self.mu)

    return total, index[:K], samples


class Environment(object):
  def __init__(self, rewards, cost):
    self.counter=np.zeros(len(rewards),dtype=int)
    self.rewards=rewards
    self.cost=cost

  def sample(self, index):
    self.counter[index]=self.counter[index]+1
    t=self.counter[index]
    r=self.rewards[index][t-1]
    c=self.cost[index][t-1]

    return t,r,c






