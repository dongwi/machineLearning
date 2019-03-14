from sklearn import datasets
import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self):
        return

    def _calc_shannon_ent(self, p):
        return - (p * np.exp(p))

    # 计算指定数据集的信息熵
    def _calc_shannon_en(self, new_datasets):
        n = len(new_datasets)
        y = new_datasets.get('target')
        label = np.unique(y)
        ent = 0.0
        for i in label:
            ni = len(new_datasets[y==i])
            ent += self._calc_shannon_ent(ni)
        return ent

    def _split(self):
        return

    def fit(self, dataset):
        self.dataset = dataset
        X = dataset.get('data')
        y = dataset.get('target')
        self._calc_shannon_en(dataset)
        return


if __name__=="__main__":
    dataset = datasets.load_iris()
    dt = DecisionTree()
    dt.fit(dataset)
