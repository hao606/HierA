
import numpy as np
import frequency_estimation as fe


class Harmony:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def set_epsilon(self, value):
        epsilon = self.epsilon
        return epsilon

    @staticmethod
    def eps2p(epsilon):
        return np.e ** epsilon / (np.e ** epsilon + 1)

    @staticmethod
    def discrete(value):
        if not -1 <= value <= 1:
            raise Exception('The value({}) is out of range'.format(value))
        p = (value + 1) / 2
        rnd = np.random.random()
        return 1 if rnd < p else -1

    def perturb(self, epsilon, value):
        rnd = np.random.random()
        return value if rnd < self.eps2p(epsilon) else -value

    def encode(self, value):
        if not -1 <= value <= 1:
            raise Exception('The value({}) is out of range'.format(value))
        epsilon = self.set_epsilon(value)
        value = self.discrete(value)
        value = self.perturb(epsilon, value)
        c = (np.e ** epsilon + 1) / (np.e ** epsilon - 1)
        return c * value

    @staticmethod
    def aggregate(perturbed_data: np.ndarray):

        pass


class PiecewiseMechanism:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def set_epsilon(self, value):
        epsilon = self.epsilon
        return epsilon

    def encode(self, value):
        epsilon = self.set_epsilon(value)
        z = np.e ** (epsilon/2)
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
            if self.split_list[i] > v:
                epsilon = self.epsilon_list[i]
                break
        return epsilon

    def encode(self, v):
        epsilon = self.set_epsilon(v)
        noise = np.random.laplace(0, 2/epsilon)
        return noise+v


class MultiHarmony(Harmony):
    def __init__(self, split_list: list, epsilon):
        super().__init__(epsilon)
        self.epsilon_list = list(np.arange(epsilon, 2*epsilon, epsilon/len(split_list)))
        self.split_list = split_list

    def set_epsilon(self, value):
        epsilon = self.epsilon_list[-1]
        for i in range(len(self.split_list)):
            if self.split_list[i] > value:
                epsilon = self.epsilon_list[i]
                break
        return epsilon


class MultiPiecewiseMechanism(PiecewiseMechanism):
    def __init__(self, split_list: list, epsilon):
        super().__init__(epsilon)
        self.epsilon_list = list(np.arange(epsilon, 2*epsilon, epsilon/len(split_list)))
        self.split_list = split_list

    def set_epsilon(self, value):
        epsilon = self.epsilon_list[-1]
        for i in range(len(self.split_list)):
            if self.split_list[i] > value:
                epsilon = self.epsilon_list[i]
                break
        return epsilon


class SegDis:
    def __init__(self, split_list: list, percent=0.2):
        self.percent = percent
        self.split_list = split_list

    def dis_value(self, data):
        dis_data = len(self.split_list)
        for i in range(len(self.split_list)):
            if self.split_list[i] > data:
                dis_data = i + 1
                break
        return dis_data

    def segment(self, dataset: np.ndarray):
        f_len = int(len(dataset) * self.percent)
        f_dataset = dataset[:f_len]
        e_dataset = dataset[f_len:]
        dis = np.vectorize(self.dis_value)
        f_dataset = dis(f_dataset)
        return f_dataset, e_dataset
        pass


class GrrHarmony(SegDis):
    def __init__(self, epsilon, split_list: list, percent=0.2, amplification=2):
        super().__init__(split_list, percent)
        self.epsilon = epsilon
        self.amplification = amplification
        pass

    def fre_est(self, dataset: np.ndarray):
        values = list(range(1, len(self.split_list)+1))
        GRR = fe.GeneralizedRandomResponse(self.epsilon, domain=values)
        grr = np.vectorize(GRR.encode)
        per_data = grr(dataset)
        f_vector = [f[1] for f in GRR.aggregate(per_data)]
        return f_vector

    def encode(self, dataset: np.ndarray):
        """
        输入为所有用户的数据集，最小隐私预算，隐私预算放大倍数
        """
        # 数据离散和抽样处理
        f_dataset, e_dataset = self.segment(dataset)

        # 估计分布频率并确定各区间的隐私预算
        f_vector = self.fre_est(f_dataset)
        # print(f_vector)
        epsilon_list = [1+(self.amplification - 1)*f*self.epsilon for f in f_vector]
        # print(epsilon_list)

        # 根据计算得出的分布频率，利用Harmony进行均值估计
        perturbed_data = []
        for value in e_dataset:
            epsilon = epsilon_list[-1]
            for i in range(len(self.split_list)):
                if self.split_list[i] > value:
                    epsilon = epsilon_list[i]
                    break
            perturbed_data.append(Harmony(epsilon).encode(value))
        return np.array(perturbed_data)
        pass


class GrrPM(SegDis):
    def __init__(self, epsilon, split_list: list, percent=0.2, amplification=2):
        super().__init__(split_list, percent)
        self.epsilon = epsilon
        self.amplification = amplification

    def fre_est(self, dataset: np.ndarray):
        values = list(range(1, len(self.split_list)+1))
        GRR = fe.GeneralizedRandomResponse(self.epsilon, domain=values)
        grr = np.vectorize(GRR.encode)
        per_data = grr(dataset)
        f_vector = [f[1] for f in GRR.aggregate(per_data)]
        return f_vector

    def encode(self, dataset: np.ndarray):
        """
        输入为所有用户的数据集，最小隐私预算，隐私预算放大倍数
        """
        # 数据离散和抽样处理
        f_dataset, e_dataset = self.segment(dataset)

        # 估计分布频率并确定各区间的隐私预算
        f_vector = self.fre_est(f_dataset)
        # print(f_vector)
        epsilon_list = [1+(self.amplification - 1)*f*self.epsilon for f in f_vector]
        # print(epsilon_list)

        # 根据计算得出的分布频率，利用Harmony进行均值估计
        perturbed_data = []
        for value in e_dataset:
            epsilon = epsilon_list[-1]
            for i in range(len(self.split_list)):
                if self.split_list[i] > value:
                    epsilon = epsilon_list[i]
                    break
            perturbed_data.append(PiecewiseMechanism(epsilon).encode(value))
        return np.array(perturbed_data)
    pass


class OueHarmony(SegDis):
    def __init__(self, epsilon, split_list: list, percent=0.2, amplification=2):
        super().__init__(split_list, percent)
        self.epsilon = epsilon
        self.amplification = amplification
        pass

    def fre_est(self, dataset: np.ndarray):
        values = list(range(1, len(self.split_list)+1))
        OUE = fe.OUE(epsilon=1, values=values)
        oue = OUE.encode
        per_data = np.empty((1, len(values)), dtype='u1')
        for v in dataset:
            per_data = np.append(per_data, [oue(v)], axis=0)
        per_data = np.delete(per_data, 0, axis=0)
        f_vector = [f[1] for f in OUE.aggregate(per_data)]
        return f_vector

    def encode(self, dataset: np.ndarray):
        """
        输入为所有用户的数据集，最小隐私预算，隐私预算放大倍数
        """
        # 数据离散和抽样处理
        f_dataset, e_dataset = self.segment(dataset)

        # 估计分布频率并确定各区间的隐私预算
        f_vector = self.fre_est(f_dataset)
        # print(f_vector)
        epsilon_list = [1+(self.amplification - 1)*f*self.epsilon for f in f_vector]
        # print(epsilon_list)

        # 根据计算得出的分布频率，利用Harmony进行均值估计
        perturbed_data = []
        for value in e_dataset:
            epsilon = epsilon_list[-1]
            for i in range(len(self.split_list)):
                if self.split_list[i] > value:
                    epsilon = epsilon_list[i]
                    break
            perturbed_data.append(Harmony(epsilon).encode(value))
        return np.array(perturbed_data)
        pass
    pass


class OuePM(SegDis):
    def __init__(self, epsilon, split_list: list, percent=0.2, amplification=2):
        super().__init__(split_list, percent)
        self.epsilon = epsilon
        self.amplification = amplification
        pass

    def fre_est(self, dataset: np.ndarray):
        values = list(range(1, len(self.split_list)+1))
        OUE = fe.OUE(epsilon=1, values=values)
        oue = OUE.encode
        per_data = np.empty((1, len(values)), dtype='u1')
        for v in dataset:
            per_data = np.append(per_data, [oue(v)], axis=0)
        per_data = np.delete(per_data, 0, axis=0)
        f_vector = [f[1] for f in OUE.aggregate(per_data)]
        return f_vector

    def encode(self, dataset: np.ndarray):
        """
        输入为所有用户的数据集，最小隐私预算，隐私预算放大倍数
        """
        # 数据离散和抽样处理
        f_dataset, e_dataset = self.segment(dataset)

        # 估计分布频率并确定各区间的隐私预算
        f_vector = self.fre_est(f_dataset)
        # print(f_vector)
        epsilon_list = [1+(self.amplification - 1)*f*self.epsilon for f in f_vector]
        # print(epsilon_list)

        # 根据计算得出的分布频率，利用Harmony进行均值估计
        perturbed_data = []
        for value in e_dataset:
            epsilon = epsilon_list[-1]
            for i in range(len(self.split_list)):
                if self.split_list[i] > value:
                    epsilon = epsilon_list[i]
                    break
            perturbed_data.append(PiecewiseMechanism(epsilon).encode(value))
        return np.array(perturbed_data)
        pass
    pass
    pass


if __name__ == '__main__':
    from data_set import GenerateData as Gd

    user_data = Gd(size=10**6).uniform()
    split_list = list(np.arange(-0.6, 1.1, 0.4))
    epsilon_list = list(np.arange(0.5, 2.6, 0.5))

    laplace = np.vectorize(Laplace(epsilon_list[1]).encode)
    multi_laplace = np.vectorize(MultiLaplace(split_list=split_list, epsilon_list=epsilon_list).encode)

    m_true = user_data.mean()
    m1_list = []
    m2_list = []
    for i in range(100):
        mean1 = laplace(user_data).mean()
        m1_list.append(mean1)
        mean2 = multi_laplace(user_data).mean()
        m2_list.append(mean2)

    m1_est = sum(abs(np.array(m1_list)-m_true)) / len(m1_list)
    m2_est = sum(abs(np.array(m2_list)-m_true)) / len(m2_list)
    print('Laplace\'s mean:', m1_est)
    print(m1_list)
    print('MultiLaplace\'s mean:', m2_est)
    print(m2_list)

    pass
