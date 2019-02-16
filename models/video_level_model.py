"""
对frame feature的降维过程建立模型
"""
# import ipdb
import time
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import DefaultConfig
from models.pca_model import PcaModel
from torch.utils.data import DataLoader
from data.dataset import VideoSet
from models.moe_model import MoeModel
from utils.utils import print_time


class StatisticModel(nn.Module):
    def __init__(self, optim, input_size):
        super(StatisticModel, self).__init__()
        self.PCA_state = True  # PCA模型训练状态，True表示已经训练，False表示未训练
        self.opt = optim
        self.pca = PcaModel(n_components=self.opt.n_components)
        if os.path.exists(self.opt.incremental_pca_params_path):
            self.pca.load(filename=self.opt.incremental_pca_params_path)
        else:
            print("PCA模型还未初始化，请先预训练PCA模型")
            self.PCA_state = False
        if self.opt.use_two_level_model:
            self.Moe1 = MoeModel(input_size=input_size,
                                 vocab_size=self.opt.vocab_size,
                                 num_mixtures=self.opt.num_mixtures)
            self.Moe2 = MoeModel(input_size=input_size,
                                 vocab_size=self.opt.vocab_size,
                                 num_mixtures=self.opt.num_mixtures)
            self.gate_fc = nn.Linear(input_size, self.opt.vocab_size*2)
        else:
            self.Moe = MoeModel(input_size=input_size,
                                vocab_size=self.opt.vocab_size,
                                num_mixtures=self.opt.num_mixtures)
        if self.opt.use_context_gate:
            # 注意，这里要用nn.Sequential而不是nn.ModuleList
            self.context_gate = nn.Sequential(
                nn.Linear(self.opt.vocab_size, self.opt.vocab_size),
                nn.Sigmoid()
            )

    def forward(self, x):
        # x的维度是[batch, input_size]
        # 提取统计特征
        if self.opt.use_gpu:
            x = x.cpu()
        x = self.statistic_feature_extraction(x)
        # PCA降维
        x = torch.from_numpy(self.pca.transform(x)).float()
        # 传入MOE网络中进行分类
        if self.opt.use_gpu:
            x = x.cuda()
        if self.opt.use_two_level_model:
            Moe1_x = self.Moe1(x)  # [batch, vocab_size]
            Moe1_x = Moe1_x.view(-1, 1)  # [batch*vocab_size, 1]
            Moe2_x = self.Moe2(x)  # [batch, vocab_size]
            Moe2_x = Moe2_x.view(-1, 1)  # [batch*vocab_size, 1]
            Moe_x = torch.cat((Moe1_x, Moe2_x), 1)  # [batch*vocab_size, 2]
            gate_x = self.gate_fc(x)  # [batch, vocab_size*2]
            gate_x = gate_x.view(-1, 2)  # [batch*vocab_size, 2]
            gate_x = F.softmax(gate_x, dim=1)
            x = torch.sum(Moe_x * gate_x, 1).view(-1, self.opt.vocab_size)
        else:
            x = self.Moe(x)
        if self.opt.use_context_gate:
            context_gate = self.context_gate(x)
            x = context_gate * x
        return x

    def statistic_feature_extraction(self, model_input):
        """
        对输入视频的帧数据提取统计特征和topk值特征
        :param model_input: 输入是tensor变量，维度是[batch_size, num_frames, num_features]
        :return: 输出是numpy变量，维度是[batch_size, (k+2)*num_features]
        """
        batch_size = model_input.shape[0]
        mean_x = torch.mean(model_input, dim=1, keepdim=True).numpy()
        var_x = torch.var(model_input, dim=1, keepdim=True).numpy()
        topk_x = torch.topk(model_input, self.opt.k, dim=1)[0].numpy()
        output = np.concatenate((mean_x, var_x, topk_x), axis=1)
        return output.reshape([batch_size, -1])

    def pretrainpca(self):
        print("预训练开始！")
        trainset = VideoSet(self.opt, state='train')
        trainloader = DataLoader(trainset,
                                 batch_size=2048,
                                 shuffle=True,
                                 num_workers=self.opt.num_workers,
                                 drop_last=False)
        times = []
        start = time.time()
        times.append(start)
        for ii, (train_input, _) in enumerate(trainloader):
            time1 = time.time()
            print_time(times[-1], time1, "第%d次迭代训练的数据读取" % (ii + 1))
            train_input = self.statistic_feature_extraction(train_input)
            print("开始第%d次迭代训练" % (ii + 1))
            self.pca.fit(train_input)
            time2 = time.time()
            print("第%d次迭代训练完成" % (ii + 1), end="，")
            print_time(time1, time2)
            times.append(time1)
            times.append(time2)
            print()
        print_time(times[0], times[-1], "整个PCA模型训练过程")
        self.pca.save(self.opt.incremental_pca_params_path)
        self.PCA_state = True

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    '''
    opt = DefaultConfig()
    dataset = VideoSet()
    dataloader = DataLoader(dataset,
                            batch_size=4096,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)
    for i, (xx, _) in enumerate(dataloader):
        print(xx.shape)
        if i == 2:
            break
    '''
    # model1 = StatisticModel()
    # model1.pretrainpca()
    opt = DefaultConfig()
    model = StatisticModel(512, opt)
    print(list(model.children()))
