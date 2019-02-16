"""
主文件，训练，验证和测试程序的入口，可通过不同的命令来指定不同的操作和参数
使用方法：python main.py train
               --train_data_root=...
               ...
"""
import os
import models
import torch
import ipdb
import time
from data.dataset import VideoSet
from config import DefaultConfig
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.utils import print_time


def train(**kwargs):
    # 根据命令行参数更新配置
    opt = DefaultConfig()
    opt.parse(kwargs)
    print("参数配置完成")

    # step1: 模型
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_num
    model = getattr(models, opt.model)(opt, 1024)  # TODO:1024这个值后期可能需要用变量代替
    if opt.model == "StatisticModel" and model.PCA_state is False:
        model.pretrainpca()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    print("模型加载完成")

    # step2: 数据
    train_data = VideoSet(opt, state='train')
    train_dataloader = DataLoader(train_data,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    print("数据集准备就绪")

    # step3: 目标函数和优化器
    criterion = torch.nn.BCELoss(size_average=False)
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=opt.weight_decay)

    # step4: 统计指标

    print("开始训练")
    # 训练
    start = time.time()
    for epoch in range(opt.max_epoch):
        epoch_start = time.time()
        t1 = time.time()
        for ii, (data, label) in enumerate(train_dataloader):
            # 训练模型参数
            input_data = Variable(data)
            target = Variable(label)
            # ipdb.set_trace()
            if opt.use_gpu:
                input_data = input_data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input_data)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            if (ii + 1) % 100:
                t2 = time.time()
                print('-------------------------------------------------')
                print('第%d个epoch，第%d个batch的loss为%.4f' % (epoch+1, ii+1, loss))
                print_time(t1, t2, '该batch训练')
                print('-------------------------------------------------')
                t1 = time.time()
        epoch_end = time.time()
        print_time(epoch_start, epoch_end, '第%d个epoch' % (epoch+1))
    end = time.time()
    print_time(start, end)

    # 最后保存一次模型参数
    model.save()


if __name__ == "__main__":
    train()
