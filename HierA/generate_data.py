import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_set import GenerateData as Gd
data_volume = 10**5

pd.Series(Gd(size=data_volume).uniform(low=-1, high=1)).to_csv('uniform_data.csv', index=False, header=False)
pd.Series(Gd(size=data_volume).normal(loc=0.6, scale=0.1)).to_csv('normal_data.csv', index=False, header=False)
pd.Series(Gd(size=data_volume).exponent(scale=0.5)).to_csv('exponent_data.csv', index=False, header=False)
pd.read_csv('adult dataset.csv', header=None)[0].to_csv('real_data.csv', index=False, header=False)

