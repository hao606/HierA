import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from frequency_estimation import GeneralizedRandomResponse as Grr
from mean_estimation import Harmony, MultiLaplace
from mean_estimation import PiecewiseMechanism as Pm

from data_set import GenerateData as Gd


def e2p(epsilon):
    p = np.e**epsilon / (np.e**epsilon+1)
    return p


class User:
    """
    用户端的扰动
    """
    def __init__(self, split_list: list, epsilon_list):
        self.split_list = split_list
        self.epsilon_list = epsilon_list
        self.k = len(epsilon_list)

    def find_level(self, v: float) -> int:
        """
        找到用户数据的隐私等级
        :param v:用户的数据
        :return: 用户所在的组区间对应的隐私等级
        """
        t = len(self.split_list)
        for i in range(len(self.split_list)):
            if self.split_list[i] > v:
                t = i+1
                break
        return t

    def per_level(self, v: float) -> int:
        """
        对用户数据的真实隐私等级进行扰动
        :param v: 用户数据
        :return: 扰动后的隐私等级
        """
        t = self.find_level(v)
        grr = Grr(epsilon=self.epsilon_list[t-1], domain=list(range(1, self.k+1)))
        _t = grr.encode(t)
        return _t

    def lpp(self, v) -> tuple:
        """
        使用Harmony算法对用户数据进行离散和扰动
        :param v: 用户数据
        :return: 扰动后的数据，1或者-1
        """
        t_ = self.per_level(v)
        epsilon = self.epsilon_list[t_-1]
        harmony = Harmony(epsilon)
        v_ = harmony.discrete(v)
        v_ = harmony.perturb(epsilon, v_)
        return tuple([t_, v_])


class Aggregator:
    """
    收集端的处理
    """
    def __init__(self, split_list: list, epsilon_list: list, un: int):
        """
        :param split_list:
        :param epsilon_list:
        :param un: 合并相邻数据集的个数
        """
        self.split_list = split_list
        self.epsilon_list = epsilon_list
        self.k = len(split_list)
        self.un = un

    def classify(self, dataset: list) -> dict:
        """
        将用户数据根据隐私等级进行分类，用字典进行保存，
        之后分别保存在level+’t‘.csv文件中
        :param dataset:用户发送给收集方的数据集，其中元素是包含用户隐私等级和用户数据的一个元组
        :return:返回为按隐私等级分类的用户数据集合
        """
        n_data = {}
        for n in range(1, len(self.epsilon_list)+1):
            n_data[n] = []
        for tup in dataset:
            n_data[tup[0]].append(tup[1])

        # for i, _data in n_data.items():
        #     pd.Series(_data).to_csv('level%d.csv' % i, index=False, header=False)
        #
        return n_data

    def data_recycle(self, n_dataset: dict) -> dict:
        """
        将低等级数据转换成高等级数据
        :param n_dataset:按隐私等级分好类的用户数据集
        :return:隐私等级转换后的用户数据集，其中每个等级的数据集都被转换成了比他高一级数据集
        """
        data_matrix = {}
        for n, dataset in n_dataset.items():
            if n == self.k:
                break
            data_matrix[n] = [dataset]
            epsilon_i = self.epsilon_list[n-1]
            p_i = e2p(epsilon_i)
            for m in range(n+1, self.k+1):
                epsilon_j = self.epsilon_list[m-1]
                p_j = e2p(epsilon_j)
                q = (p_i+p_j-1) / (2*p_i-1)
                dataset_ = []
                for v in dataset:
                    rnd = np.random.random()
                    v_ = v if rnd < q else -v
                    dataset_.append(v_)
                data_matrix[n].append(dataset_)
            data_matrix[self.k] = [n_dataset[self.k]]
        # pd.DataFrame(data_matrix).to_csv('recycled data.csv', index=False, header=False)

        return data_matrix

    def union(self, datamatrix: dict) -> dict:
        """
        将经过隐私等级转换的数据集中的相同隐私等级数据进行合并，得到V*
        :param datamatrix: 隐私等级转换后的用户数据集
        :return: 合并之后的用户数据集
        """
        dataset = dict()

        # 合并相邻数据集
        # dataset[1] = datamatrix[1][0]
        # for i in range(2, self.k+1):
        #     dataset[i] = datamatrix[i][0] + datamatrix[i-1][1]
        #
        # dataset[self.k] = dataset[self.k]+datamatrix[self.k][0]

        m = self.un
        i = 1
        while i+m-1 <= self.k:
            dataset[i+m-1] = []
            for j in range(i, i+m):
                dataset[i+m-1].extend(datamatrix[j][i+m-1-j])
            # pd.Series(dataset[i+m-1]).to_csv('union data l%d.csv' % (i+m-1), index=False, header=False)
            i += m

        if i <= self.k:
            dataset[self.k] = []
            for j in range(i, self.k+1):
                dataset[self.k].extend(datamatrix[j][self.k-j])
        # for i in range(1, self.k+1):
        #     pd.Series(dataset[i]).to_csv('union data lv%d.csv' %i, index=False, header=False)

        return dataset

    def average(self, dataset: list) -> float:
        """
        计算每个隐私等级的数据集中-1和1的个数，并进行校正之后进行求和进而求平均值。
        :param dataset: 收集方收集到的用户数据集，其中每个元素是包含用户隐私等级和用户数据的一个元组
        :return: 估计均值mean
        """
        n_dataset = self.classify(dataset)
        datamatrix = self.data_recycle(n_dataset)
        dataset = self.union(datamatrix)
        s = 0
        num = 0
        for j, data in dataset.items():
            n = len(data)
            n_1 = data.count(1)
            n_2 = data.count(-1)

            p_j = e2p(self.epsilon_list[j-1])
            n1 = (p_j-1)*n / (2*p_j-1) + n_1 / (2*p_j-1)
            n2 = (p_j-1)*n / (2*p_j-1) + n_2 / (2*p_j-1)
            if not 0 <= n1 <= n:
                n1 = 0 if n1 < 0 else n
            if not 0 <= n2 <= n:
                n2 = 0 if n2 < 0 else n

            num += n
            s += n1-n2
        mean = s / num
        # print('s=', s)
        # print('num=', num)
        return mean


def set_par(set_num=5, epsilon=0.5, epsilon_amplify=5):
    """
    这里我们默认用户数据值域为[-1,1]，子区间和对应的隐私等级均按等差数列确定。
    :param set_num:子区间划分数量
    :param epsilon:隐私保护等级最高的隐私预算，即最小的隐私预算
    :param epsilon_amplify: 隐私预算放大倍数
    :return:子区间分割点：list，隐私预算集合：list
    """
    split_list = list(np.arange(-1+2/set_num, 1.1, 2/set_num))
    epsilon_list = list(np.arange(epsilon, epsilon*epsilon_amplify+0.0001, epsilon*(epsilon_amplify-1)/(set_num-1)))
    epsilon_list.reverse()

    # print('split list:', split_list)
    # print('epsilon_list:', epsilon_list)
    return split_list, epsilon_list

    pass


def hiera(userdata, set_num=5, epsilon=0.2, epsilon_amplify=5, un=2):
    split_list, epsilon_list = set_par(set_num, epsilon, epsilon_amplify)
    lpp = User(split_list, epsilon_list).lpp
    average = Aggregator(split_list, epsilon_list, un=un).average

    # pd.Series(userdata).to_csv('original userdata.csv', index=False)
    send_dataset = []
    for v in userdata:
        send_dataset.append(lpp(v))
    # pd.DataFrame(send_dataset).to_csv('perturbed userdata.csv', index=False, header=False)

    mean = average(send_dataset)

    return mean


def cmp1(userdata: np.array, epsilons, set_num=5, epsilon_amplify=5, un=2) -> dict:
    """
    横向对比
    在仅仅改变epsilon的情况下，比较Harmony，PM，Laplace方法的AE和MSE。
    :param userdata: 用户原始数据集
    :param set_num: 子区间划分个数
    :param epsilons: 隐私预算变化集合
    :param epsilon_amplify: 最大隐私预算与最小隐私预算的比值
    :param un: 合并相邻数据集的个数
    :return :返回不同方法在不同隐私预算下的的mae和mse数组
    """
    mae = {'harmony': [], 'pm': [], 'laplace': [], 'hiera': []}
    mse = {'harmony': [], 'pm': [], 'laplace': [], 'hiera': []}
    methods = ('harmony', 'pm', 'laplace', 'hiera')
    m_true = userdata.mean()

    for epsilon in epsilons:
        ae_1, ae_2, ae_3, ae_4 = [], [], [], []
        se_1, se_2, se_3, se_4 = [], [], [], []
        harmony = np.vectorize(Harmony(epsilon=epsilon).encode)
        pm = np.vectorize(Pm(epsilon=epsilon).encode)
        split_list, epsilon_list = set_par(set_num=set_num, epsilon=epsilon, epsilon_amplify=epsilon_amplify)
        multi_laplace = np.vectorize(MultiLaplace(split_list, epsilon_list).encode)

        for i in range(100):
            m_harmony = harmony(userdata).mean()
            m_pm = pm(userdata).mean()
            m_laplace = multi_laplace(userdata).mean()
            m_hiera = hiera(userdata, set_num=5, epsilon=epsilon, epsilon_amplify=5, un=un)
            for m_, ae_ in zip((m_harmony, m_pm, m_laplace, m_hiera), (ae_1, ae_2, ae_3, ae_4)):
                ae_.append(abs(m_ - m_true))

            for m_, se_ in zip((m_harmony, m_pm, m_laplace, m_hiera), (se_1, se_2, se_3, se_4)):
                se_.append((m_-m_true)**2)

        for method, ae in zip(methods, (ae_1, ae_2, ae_3, ae_4)):
            mae[method].append(sum(ae) / len(ae))

        for method, se in zip(methods, (se_1, se_2, se_3, se_4)):
            mse[method].append(sum(se) / len(se))
    return mae, mse
    pass


def cmp2(userdata: np.array, set_num: int, epsilons: list, epsilon_amplifies: list, uns: list) -> dict:
    """
    纵向对比,分别改变隐私放大程度EA、相邻数据集的合并个数un，观察其对HierA方法的影响。观察HierA方法进行
    均值估计时的MAE
    :param userdata: 用户数据集
    :param set_num: 数据值域子区间个数，即隐私等级数
    :param epsilons: 变化的隐私预算集合
    :param epsilon_amplifies: 隐私放大倍数集合
    :param uns: 相邻数据集合并个数
    :return: 不同隐私放大倍数，和不同数据集合并个数，随最小隐私预算变化的集合。
    """
    m_true = userdata.mean()
    mae_uns = {i: [] for i in uns}

    # mae_eas = {i: [] for i in epsilon_amplifies}
    # for ea in epsilon_amplifies:
    #     ae_ea = []
    #     for epsilon in epsilons:
    #         ae = []
    #         for i in range(100):
    #             m = hiera(userdata, set_num=set_num, epsilon=epsilon, epsilon_amplify=ea)
    #             ae.append(abs(m - m_true))
    #         ae_ea.append(sum(ae) / len(ae))
    #     mae_eas[ea] = ae_ea

    for un in uns:
        ae_un = []
        for epsilon in epsilons:
            ae = []
            for i in range(100):
                m = hiera(userdata, set_num=set_num, epsilon=epsilon, epsilon_amplify=5, un=un)
                ae.append(abs(m - m_true))
            ae_un.append(sum(ae) / len(ae))
        mae_uns[un] = ae_un
    return mae_uns

    pass



