from sklearn import datasets
import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self):
        return

    # 计算信息熵
    def _calc_shannon_ent(self, p):
        return - (p * np.exp(p))

    # 计算指定数据集的信息熵
    def _calc_shannon_en(self, y):
        label = np.unique(y)
        n = len(y)
        ent = 0.0
        for i in label:
            ni = len(y[y == i])
            ent += self._calc_shannon_ent(ni / n)
        return ent

    # 计算指定特征的信息熵
    def _split_feat_ent(self, feat_index, X, y):
        n = X.shape[0]
        feat = X[:, feat_index]
        feat_unique = np.unique(feat)
        ent = 0.0
        for f in feat_unique:
            y_f = y[feat == f]
            y_f_n = len(y_f)
            ent += (float(y_f_n) / n) * self._calc_shannon_en(y_f)
        return ent

    #
    def _pick_feat(self, X, y):
        min_feat_index = np.inf
        min_ent = np.inf
        for i in range(X.shape[1]):
            ient = self._split_feat_ent(i, X, y)
            if min_ent > ient:
                min_ent = ient
                min_feat_index = i
        return min_feat_index


    # 适配数据
    def fit(self):
        X = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [1, 0, 0, 0],
                      [2, 1, 0, 0],
                      [2, 2, 1, 0],
                      [2, 2, 1, 1],
                      [1, 2, 1, 1]])
        y = np.array(['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y'])
        self._pick_feat(X, y)
        return


if __name__ == "__main__":
    dt = DecisionTree()
    dt.fit()
