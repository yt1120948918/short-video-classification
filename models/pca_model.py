"""
利用sklearn构建pca模型
"""

import pickle
# import torch
# import torch.nn as nn
# import numpy as np
from sklearn.decomposition import IncrementalPCA
# from sklearn.preprocessing import normalize


class PcaModel:
    def __init__(self, n_components=1024, n_samples_seen=None, components=None, singular_values=None,
                 mean=None, var=None, explained_variance=None, explained_variance_ratio=None, noise_variance=None):
        self.components = components
        self.n_samples_seen = n_samples_seen
        self.singular_values = singular_values
        self.mean = mean
        self.var = var
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio
        self.noise_variance = noise_variance
        self.incremental_pca = IncrementalPCA(n_components=n_components, whiten=True)
        if self.components and self.mean:
            self.incremental_pca.n_samples_seen_ = self.n_samples_seen
            self.incremental_pca.components_ = self.components
            self.incremental_pca.singular_values_ = self.singular_values
            self.incremental_pca.mean_ = self.mean
            self.incremental_pca.var_ = self.var
            self.incremental_pca.explained_variance_ = self.explained_variance
            self.incremental_pca.explained_variance_ratio_ = self.explained_variance_ratio
            self.incremental_pca.noise_variance_ = self.noise_variance

    def fit(self, x):
        # 注意这里输入的x为numpy变量
        self.incremental_pca.partial_fit(x)

    def transform(self, x):
        # 注意这里输入的x为numpy变量
        x = self.incremental_pca.transform(x)
        # TODO:需要加入L2 normalize，尝试过下面normalize代码，但是有点问题
        # x = normalize(x, norm='l2)
        return x

    def load(self, filename):
        params = pickle.load(open(filename, 'rb'))
        self.incremental_pca.n_samples_seen_ = params['n_samples_seen']
        self.incremental_pca.components_ = params['components']
        self.incremental_pca.singular_values_ = params['singular_values']
        self.incremental_pca.mean_ = params['mean']
        self.incremental_pca.var_ = params['var']
        self.incremental_pca.explained_variance_ = params['explained_variance']
        self.incremental_pca.explained_variance_ratio_ = params['explained_variance_ratio']
        self.incremental_pca.noise_variance_ = params['noise_variance_']

    def save(self, filename):
        params = dict()
        params['n_samples_seen'] = self.incremental_pca.n_samples_seen_
        params['components'] = self.incremental_pca.components_
        params['singular_values'] = self.incremental_pca.singular_values_
        params['mean'] = self.incremental_pca.mean_
        params['var'] = self.incremental_pca.var_
        params['explained_variance'] = self.incremental_pca.explained_variance_
        params['explained_variance_ratio'] = self.incremental_pca.explained_variance_ratio_
        params['noise_variance_'] = self.incremental_pca.noise_variance_
        pickle.dump(params, open(filename, 'wb'))
