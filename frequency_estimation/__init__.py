

import numpy as np
import random


class RandomResponse:
    def __init__(self, p):
        self.p = p

    def encode(self, value):
        perturbed_value = 0 if value == 1 else 1
        result = value if np.random.binomial(1, self.p) else perturbed_value

        return result


class GeneralizedRandomResponse:
    def __init__(self, epsilon, domain: list):
        self.domain = domain
        self.p = np.e ** epsilon / (np.e ** epsilon + len(self.domain) - 1)
        self.q = 1 / (np.e ** epsilon + len(self.domain) - 1)

    def encode(self, value):
        values = self.domain.copy()
        values.remove(value)
        result = value if np.random.binomial(1, self.p) == 1 else random.choice(values)
        return result

    def aggregate(self, perturbed_data: np.ndarray):
        f_vector = {}
        for c in self.domain:
            _f = sum(perturbed_data == c) / len(perturbed_data)
            f = (_f - self.q) / (self.p - self.q)
            f = 0 if f < 0 else f
            f_vector[c] = f
        return sorted(f_vector.items(), key=lambda x: x[0])


class UnaryEncoding:
    def __init__(self, p, q, values: list):
        self.p = p
        self.q = q
        self.values = values

    def unify(self, value):
        unary_list = np.zeros(len(self.values), dtype='u1')
        index = self.values.index(value)
        unary_list[index] = 1
        return unary_list

    def encode(self, value):
        rr1 = RandomResponse(self.p)
        rr0 = RandomResponse(1-self.q)
        bit_vector = self.unify(value)

        for i in range(len(bit_vector)):
            if bit_vector[i] == 1:
                bit_vector[i] = rr1.encode(1)
            elif bit_vector[i] == 0:
                bit_vector[i] = rr0.encode(0)
        return bit_vector

    def aggregate(self, perturbed_data: np.ndarray):
        f_vector = {}
        for i in range(len(self.values)):
            _f = sum(perturbed_data[:, i] == 1) / len(perturbed_data)
            f = (_f - self.q) / (self.p - self.q)
            f = 0 if f < 0 else f
            f_vector[self.values[i]] = f
        return sorted(f_vector.items(), key=lambda x: x[0])


class RAPPOR(UnaryEncoding):
    def __init__(self, epsilon, values: list):
        t = np.e ** (epsilon/2)
        p = t / (t+1)
        q = 1 - p
        super().__init__(p, q, values)


class OUE(UnaryEncoding):
    def __init__(self, epsilon, values: list):
        p = 1 / 2
        q = 1 / (np.e**epsilon + 1)
        super().__init__(p, q, values)


