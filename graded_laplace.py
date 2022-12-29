import numpy as np


class Laplace:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def encode(self, v):
        noise = np.random.laplace(0, 2/self.epsilon)
        return v+noise


class MultiLaplace:
    def __init__(self, split_list: list, epsilon_list: list):
        self.epsilon_list = epsilon_list
        self.split_list = split_list

    def set_epsilon(self, v):
        epsilon = self.epsilon_list[-1]
        for i in range(len(self.split_list)):
            if v < self.split_list[i]:
                epsilon = self.epsilon_list[i-1]
                break
        return epsilon

    def encode(self, v):
        epsilon = self.set_epsilon(v)
        noise = np.random.laplace(0, 2/epsilon)
        return noise+v


