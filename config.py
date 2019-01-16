"""
配置文件，所有可配置的变量都集中在此，并提供默认值
"""
import warnings


class DefaultConfig:
    # 基本超参数
    batch_size = 8  # 每次读取多少个视频文件

    # Window下DataLoader不支持多进程处理，所以这里设置为0，在Linux下可以设置为其他数值
    num_workers = 0  # 读取数据时所用的线程数量

    use_gpu = True  # 是否使用GPU
    gpu_num = "1"  # GPU型号，"0"或者"1"
    load_model_path = None  # 所用模型参数文件路径 r'./checkpoints/model.pth'
    lr = 0.01  # 学习率
    weight_decay = 1e-4  # 优化器中的参数weight decay
    max_epoch = 8  # 最大循环训练次数
    model = 'StatisticModel'  # 还可以填写'DBoFModel'、'NetVLADModel'、'FVNetModel'和'RNNModel'

    # 视频抽帧参数
    every_ms = 1000  # 帧之间时间间隔，单位ms
    cut_time = 1  # 截取末尾，单位s
    max_num_frames = 300  # 每个视频最多采样数
    frames_num = 16  # 每个视频最终的帧数

    # 训练集和验证集的data路径
    train_data_path = r'D:\data\video_classification\train_set\train_video'
    validation_data_path = r'D:\data\video_classification\validation_set\validation_video'
    # 测试集的data路径
    test_data_path = r''

    # 训练集和验证集的label路径
    train_label_path = r'D:\data\video_classification\train_set\train_label.json'
    validation_label_path = r'D:\data\video_classification\validation_set\validation_label.json'

    # 提取帧图片特征的CNN网络的相关参数
    feature_extraction_model = 'vgg19'  # 还可以填写'inceptionv3'和'resnet50'
    train_size = 256  # 训练集的resize
    test_size = 224  # 验证集和测试集的resize
    crop_size = 224  # 裁剪的size

    # MOE分类网络参数
    vocab_size = 63  # 总标签数量，比赛中已经给出是63种
    num_mixtures = 4  # 单层MoE网络中专家数量
    use_context_gate = True
    use_two_level_model = True

    # 方法一参数
    k = 6  # topk的k值
    n_components = 1024  # PCA降维后的维度
    incremental_pca_params_path = r'./models/incremental_pca_params.pkl'

    def parse(self, kwargs):
        """
        根据字典kwargs更新config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        # 打印配置参数
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
