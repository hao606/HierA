#  Duchi's mechanism for mean_estimation over one dimensional numerical data

import numpy as np
import matplotlib.pyplot as plt


class Harmony:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    @staticmethod
    def eps2p(epsilon):
        return np.e ** epsilon / (np.e ** epsilon + 1)

    @staticmethod
    def discrete(value):
        p = (value + 1) / 2
        rnd = np.random.random()
        return 1 if rnd < p else -1

    def perturb(self, epsilon, value):
        rnd = np.random.random()
        return value if rnd < self.eps2p(epsilon) else -value

    def encode(self, value):
        if not -1 <= value <= 1:
            raise Exception('The value({}) is out of range'.format(value))
        epsilon = self.epsilon
        value = self.discrete(value)
        value = self.perturb(epsilon, value)
        c = (np.e ** epsilon + 1) / (np.e ** epsilon - 1)
        return c * value

