import torch
import  os, glob
import numpy as np
import  random, csv
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

class Data_Set(Dataset):
    """
    pseudolabels : [0,0,...]
    images_lists: [0,1,2,...]
    """ 
    def __init__(self, video_folder, resize_height=256, resize_width=256, time_steps=3, pseudoflag=False, pseudolabels=None):
        
        super(Data_Set, self).__init__()

        self.pseudolabels = pseudolabels 
        self.pseudoflag = pseudoflag
        
        self.dir = video_folder
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        # 定义要剪取的帧的长度
        self.time_steps = time_steps
        # 进行视频信息提取
        self.setup()
        # 裁剪每个视频并将所有裁剪出的视频片段合并为一个大列表,video_seg_list为剪辑好的视频列表，每个列表中的每个元素为一个视频剪辑
        self.video_seg_list = self.video_clip()

        # 保存原始帧的大小

        temp_frame = Image.open(self.video_seg_list[0][0])
        self.original_shape = (360, 360)
        self.frame_shape = (360, 360)

    # 定义初始处理函数
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_nummber_clip(self):
        video_frame_num = []
        video_num_clip = []
        for k, v in self.videos.items():
            num_clip = len(v['frame']) - self.time_steps + 1
            video_num_clip.append(num_clip)
            video_frame_num.append(len(v['frame']))
        # 返回的是每个视频的帧数以及能提取的剪辑的个数
        return video_frame_num, video_num_clip

    # 定义裁剪视频片段的函数
    def video_clip(self):
        video_seg = []
        video_num = []
        # 遍历每个有序字典
        for k,v in self.videos.items():
            # k: 01
            # v: {'path': '../Data/ped2/training/frames/01', 'frame': ['../Data/ped2/training/frames/01/000.jpg',
            # '../Data/ped2/training/frames/01/001.jpg',。。。], 'length': 120},
            # print(k,v)
            # # 能裁剪出的帧数：len(a)-num_his+1
            num_clip = len(v['frame']) - self.time_steps + 1
            video_num.append(num_clip)
            # 遍历没个视频的帧，并进行剪辑操作 v['frame']为视频中的所有帧的列表
            for i in range(num_clip):
                # print(i)
                video_seg.append(v['frame'][i:i+self.time_steps])
        # # 打印出视频的剪辑列表
        # print(video_seg)
        # # 打印出每个视频视频剪辑数
        # print(video_num)
        # # 总的剪辑数
        # print(len(video_seg))
        # # 总的剪辑数
        # print(sum(video_num))
        return video_seg

    def __len__(self):
        return len(self.video_seg_list)

    def __getitem__(self, idx):

        if self.pseudoflag == False and self.pseudolabels == None :
            
            image_path_list = self.video_seg_list[idx]
            # 定义处理器
            tf = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),  # string path= > image data
                transforms.Resize((self._resize_height, self._resize_width)),
                transforms.ToTensor()
                # transforms.normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])

            image = []
            # 循环读取当前剪辑中的图片
            for img_path in image_path_list:
                img_sub = tf(img_path)
                image.append(img_sub)
                # print(img_sub.shape)
            # image_clip = torch.cat(image, dim=0)
            # image_clip = image_path_list
            # 返回的是一个tensor，shape为[time_steps=4*2, 256, 256]
            # 再把image_clip变为[channel, depth, h, w]的形式，方便用dataloader包装后变成[batch, channel, depth, h, w]
            # image_clip = image_clip.unsqueeze(0)
            return image

        else:
            # 对进行伪标签的训练

            image_path_list = self.video_seg_list[idx]
            # 定义处理器
            tf = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),  # string path= > image data
                transforms.Resize((self._resize_height, self._resize_width)),
                transforms.ToTensor()
                # transforms.normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])

            image = []
            # 循环读取当前剪辑中的图片
            for img_path in image_path_list:
                img_sub = tf(img_path)
                image.append(img_sub)
                # print(img_sub.shape)
            # image_clip = torch.cat(image, dim=0)
            # image_clip = image_path_list
            # 返回的是一个tensor，shape为[time_steps=4*2, 256, 256]
            # 再把image_clip变为[channel, depth, h, w]的形式，方便用dataloader包装后变成[batch, channel, depth, h, w]
            # image_clip = image_clip.unsqueeze(0)
            return image, self.pseudolabels[idx]
            



def main():
    data_set = Data_Set('../data/ped2/training/frames', resize_height=360, resize_width=360, time_steps=5)
    print('done！')
    print('共拥有的剪辑个数为：')
    print(data_set.__len__())
    # 共  个

    image = data_set.__getitem__(0)  # I_0 size is [3,360,360]
    #  print(len(image))
    # I_0, I_1 = data_set.__getitem__(0)
    #  print(I_0)
    # print(image_clip)
    # print(image_clip.shape)
    # print(image_clip.max())
    # print(image_clip.min())

    # # 进行画图测试
    # image1 = image_clip[0].numpy()
    # plt.imshow(image1, cmap='Greys_r')
    # plt.show()


    # img = Image.open('../Data/ped2/training/frames/16/142.jpg').convert('L')
    # img = np.array(img, 'f')
    # print(img)


    # 测试用装载数据
    train_loader = DataLoader(data_set, batch_size=12, shuffle=True, num_workers=32)
    x = next(iter(train_loader))
    #  print(x.shape)
    print('done')
    # x_pre, x_later = x.chunk(2, dim=2)

    # print(x_pre.shape)
    # print(x_later.shape)
    #
    # print(x)
    # print(x_pre)
    # print(x_later)


if __name__ == "__main__":
    main()
