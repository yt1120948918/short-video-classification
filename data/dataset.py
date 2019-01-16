import os
import cv2
import copy
import json
# import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from config import DefaultConfig
from torch.utils.data import DataLoader
from models.cnn_model import FeatureExtractModels

VGG_LIST = [104., 117., 123.]


class VideoSet(data.Dataset):
    """
    dataset返回的值不是cuda变量，如果保存为cuda变量会导致内存爆炸
    """
    def __init__(self, optim, transform=None, state='train'):
        super(VideoSet, self).__init__()

        self.opt = optim
        self.featureNet = FeatureExtractModels(self.opt.feature_extraction_model)
        self.featureNet.eval()  # 将featureNet的状态置为测试
        if self.opt.use_gpu:
            self.featureNet.cuda()
        self.state = state
        root = ''
        if self.state == 'train':
            root = self.opt.train_data_path
        elif self.state == 'validation':
            root = self.opt.validation_data_path
        elif self.state == 'test':
            root = self.opt.test_data_path
        self.videos = [os.path.join(root, video) for video in os.listdir(root)]

        if transform is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.state == 'test' or self.state == 'validation':
                self.transforms = transforms.Compose([
                    transforms.Resize(self.opt.test_size),
                    transforms.CenterCrop(self.opt.crop_size),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(self.opt.train_size),
                    transforms.RandomResizedCrop(self.opt.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        labels = {}
        if self.state == 'train':
            with open(self.opt.train_label_path) as f:
                labels = json.load(f)
        elif self.state == 'validation':
            with open(self.opt.validation_label_path) as f:
                labels = json.load(f)
        video_path = self.videos[index]

        if self.state == 'test':
            label = [-1] * 63
        else:
            label_key = video_path.split('\\')[-1].split('.')[0]
            label = labels[label_key]

        # 保证每个视频的帧数不多于给定帧数
        frame_data = video2frame(video_path, self.opt)
        n = frame_data.shape[0]
        if n > self.opt.frames_num:
            lst = np.random.choice(np.arange(n), self.opt.frames_num)
            frame_data = frame_data[lst]

        # 将帧数据传到CNN网络中提取特征
        for ii in range(len(frame_data)):
            batch_frame_data = Image.fromarray(frame_data[ii])
            batch_frame_data = self.transforms(batch_frame_data).unsqueeze(0)
            if self.opt.use_gpu:
                batch_frame_data = batch_frame_data.cuda()
            if ii == 0:
                frame_feature = self.featureNet(batch_frame_data).cpu().data
            else:
                batch_feature = self.featureNet(batch_frame_data).cpu().data
                frame_feature = np.concatenate((frame_feature, batch_feature), 0)

        # 这里还需要保证较短的视频有足够的帧特征
        if n < self.opt.frames_num:
            try:
                res = self.opt.frames_num // n + 1
            except ZeroDivisionError:
                print(str(self.videos[index]) + " 无法提取帧数据，请检查视频是否过短")
                raise
            frame_feature = np.tile(frame_feature, (res, 1))[:self.opt.frames_num]
        return frame_feature, np.array(label)

    def __len__(self):
        return len(self.videos)


def video2frame(filename, option):
    """
    参数
    filename:文件所在路径
    every_ms:帧之间时间间隔 单位ms
    cut_time:截取末尾，单位s
    max_num_frames:每个视频最多采样数
    """
    retrieved_frames = []
    video_cap = cv2.VideoCapture(filename)
    if not video_cap.isOpened():
        print('can not open file:'+filename)
        return
    last_ts = -99999
    num_retrieved = 0
    total_frames = video_cap.get(7)
    fps = video_cap.get(5)
    video_time = total_frames / fps
    # 限制每个视频抽取的最大帧数
    while num_retrieved < option.max_num_frames:
        # 跳过间隔时间小于阈值的帧
        while video_cap.get(0) < option.every_ms + last_ts:
            if not video_cap.read()[0]:
                return np.array(retrieved_frames)
        last_ts = video_cap.get(0)  # 更新上次抽取帧所在的时间点
        if last_ts > (video_time-option.cut_time)*1000:  # 截掉最后1秒的视频
            break
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        retrieved_frames.append(copy.copy(frame))
        num_retrieved += 1
    video_cap.release()
    return np.array(retrieved_frames)


if __name__ == '__main__':
    opt = DefaultConfig()
    dataset = VideoSet(opt)
    dataloader = DataLoader(dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)
    for i, (xx, _) in enumerate(dataloader):
        print(xx.shape)
        if i == 2:
            break
