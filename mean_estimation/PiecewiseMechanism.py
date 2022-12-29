# Wang ning's PiecewiseMechanism for mean estimation
# over one dimensional numerical data


import numpy as np


class PiecewiseMechanism:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def encode(self, value):
        z = np.e ** (self.epsilon/2)
        p1 = (value+1) / (2+2*z)
        p2 = z / (z+1)
        p3 = (1-value) / (2+2*z)

        c = (z+1) / (z-1)
        left = (c+1) * value / 2 - (c-1) / 2
        right = (c+1) * value / 2 + (c-1) / 2

        rnd = np.random.random()
        if rnd < p1:
            result = -c + np.random.random() * (left - (-c))
        elif rnd < p1+p2:
            result = (right-left) * np.random.random() + left
        else:
            result = (c-right) * np.random.random() + right
        return result


