import torch
import torch.nn as nn 
import numpy as np 
import torchvision
from alexnet import MY_Net

import clustering
from dataset import *
from  util import  UnifLabelSampler
from visdom import Visdom

viz=Visdom()

# print(torch.cuda.is_available())

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义参数
#  data = '../data/pokeman'
exp = "result/exp"
lr = 0.05
wd = -5
nmb_cluster = 2  # Kmeans的聚类个数
num_workers = 64
image_num = 5  # he number of image of a clip
batch_size_common = 8

viz.line([0.], [0.], win="loss", opts=dict(title='loss'))
viz.line([0.], [0.], win="predict", opts=dict(title='predict'))


def main():

    # 定义数据train_loader
    data_set = Data_Set('../data/ped2/testing/frames', resize_height=360, resize_width=360, time_steps=5)
    train_loader = DataLoader(data_set, batch_size=batch_size_common, shuffle=False, num_workers = num_workers)
    
    # 定义模型
    
    model_backbone = torchvision.models.resnet50(pretrained=True, progress=True).to(device)
    model_gcn = MY_Net(num_classes=nmb_cluster).to(device)

    #  boxes_features = torch.randn(1,5,1000).to(device)
    #  output = model_gcn(boxes_features)
    #  # 输出为 [1,2]
    #  print(output.shape)
    #  #  print(output.shape)
    # 创建优化器
    optimizer_gcn = torch.optim.SGD(filter(lambda p: p.requires_grad, model_gcn.parameters()),lr=lr)
    
    # 创建损失函数
    # define loss function
    criterion = nn.CrossEntropyLoss().to(device)
    # 创建聚类函数
    deepcluster = clustering.__dict__['Kmeans'](nmb_cluster)

    # 进行训练
    for epoch in range(100):
        # 对每一个epoch进行迭代
        output_total_features = []
        for batchidx, image in enumerate(train_loader, 0):

            # remove the last batch data
            if batchidx == len(train_loader) - 1:
                batch_size = image[0].shape[0]
            else:
                batch_size = batch_size_common

            # 删除model_gcn的top_layer 并切换到eval模式
            model_gcn.top_layer = None
            model_gcn.eval()
            # 得到训练集的特征， 先将backbone切换到eval模式
            model_backbone.eval()

            # 进行图像的堆叠，一次过网络，5个[batch,3,360,360] => [batch*5,3,360,360]
            image = torch.cat([image[i].to(device) for i in range(image_num)], dim=0)
            #  print(image.shape)
            # 获得图像特征 大小为[batch*5, 1000]
            image_features = model_backbone(image)
            #  print(image_features.shape)
            # 将每一张图片的特征分开，形成列表 image_features = [[batch_size,1000],[batch_size,1000],[batch_size,1000],...5次]
            image_features = [image_features[i*batch_size : i*batch_size + batch_size] for i in range(image_num)]
            #  print(len(image_features))
            # 遍历batch

            # 定义output记录图的输出
            output = []

            for batch_clip in range(batch_size):
                I0 = image_features[0][batch_clip].unsqueeze(0)
                I1 = image_features[1][batch_clip].unsqueeze(0)
                I2 = image_features[2][batch_clip].unsqueeze(0)
                I3 = image_features[3][batch_clip].unsqueeze(0)
                I4 = image_features[4][batch_clip].unsqueeze(0)
                # I.shape is [1, 5, 1000]
                I = torch.stack([I0, I1, I2, I3, I4], dim=1)

                if batchidx % 100 == 0:
                    if batch_clip == 0:
                        # 遍历几次batch就输出几次，加上判断屏蔽这种情况
                        print("data prepare done~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        print("epoch: %d , batchidx: %d" % (epoch, batchidx))

                # size is [1,5000] reference to [1167,4096]
                output_features = model_gcn(I)
                # add this features to a list

                output.append(output_features)

            # output为一个batch迭代后输出特征的列表  [batch, 5000]  # add data into disk
            output = torch.cat(output, dim=0).detach().cpu().numpy()
            output_total_features.append(output)

        output_total_features = np.concatenate(output_total_features, axis=0) # torch.cat(output_total_features, dim=0)

        # 对一个batch中的特征进行聚类分析
        # 先定义clustering_loss  [1962,500] is the size
        clustering_loss = deepcluster.cluster(output_total_features, verbose=False)

        # 释放显存
        output_features = 0
        torch.cuda.empty_cache()

        # assign pseudo-labels 分配伪标签
        #  train_dataset = clustering.cluster_assign(deepcluster.images_lists,
        #                                            dataset.imgs)
        # 实现分配伪标签的操作
        ########################################################################
        image_indexes, pseudolabels = [], []
        for cluster, images in enumerate(deepcluster.images_lists):
            image_indexes.extend(images)
            pseudolabels.extend([cluster] * len(images))

        #######################################################################################################
        # 保存聚类后的结果，避免重复计算
        # np.save('pseudolabels.npy', pseudolabels)
        # np.save('images_lists.npy', deepcluster.images_lists)

        # pseudolabels = np.load('pseudolabels.npy').tolist()
        # images_lists = np.load('images_lists.npy', allow_pickle=True).tolist()
        ######################################################################################################

        train_dataset = Data_Set('../data/ped2/testing/frames', resize_height=360, resize_width=360,
                time_steps=5, pseudoflag=True , pseudolabels=pseudolabels)

        # 对分配标签后的数据集进行检测
        #  image_data_list, image_label  = train_dataset.__getitem__(0)
        

        # uniformly sample per target 进行均匀采集数据的操作
        sampler = UnifLabelSampler(int(len(train_dataset)),
                                   deepcluster.images_lists)
        # sampler = UnifLabelSampler(int(len(train_dataset)),
        #                            images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer 设置全连接层
        # 创建带有激活层的top_layer
        model_gcn.top_layer = nn.Sequential(
                # 由于只有一个batch，调用正则层会报错
                #  nn.BatchNorm1d(1000 * 5),
                #  nn.ReLU(),
                #  nn.Linear(100*5,100*5),
                nn.ReLU(),
                nn.Linear(100*5,10*5),
                nn.ReLU(),
                nn.Linear(10*5, nmb_cluster),
                nn.Sigmoid(),
                )
        model_gcn.top_layer.to(device)


        # 进行有监督的训练
        loss = train(epoch, train_dataloader, model_backbone, model_gcn, criterion, optimizer_gcn)


def train(epoch, train_dataloader, model_backbone, model_gcn, criterion, optimizer_gcn):
    # 此时为训练一个大batch
    # 网络切换到train模式
    model_gcn.train()
    ########################################################################
    # 先不训练bacbone试试
    # 将backone网络也切换到训练模式
    model_backbone.train()
    # 创建backbone网络的优化器
    optimizer_backbone = torch.optim.SGD(
        model_backbone.parameters(),
        lr=lr,
        # weight_decay=10**wd,
    )
    # 创建model_gcn的top_layer网络的优化器
    optimizer_tl = torch.optim.SGD(
        model_gcn.top_layer.parameters(),
        lr=lr,
        # weight_decay=10**wd,
    )
    
    # 对带有标签的数据集进行遍历并训练
    # 参考    #  image_data_list, image_label  = train_dataset.__getitem__(0)
    # for i, (input_tensor, target) in enumerate(train_dataloader):

    for batchidx, (image, image_label) in enumerate(train_dataloader):

        # print(batchidx)

        # 进行图像的堆叠，一次过网络，5个[batch,3,360,360] => [batch*5,3,360,360]
        image = torch.cat([image[i].to(device) for i in range(image_num)], dim=0)
        #  print(image.shape)
        # 获得图像特征 大小为[batch*5, 1000]
        image_features = model_backbone(image)
        #  print(image_features.shape)
        # 将每一张图片的特征分开，形成列表 image_features = [[batch_size,1000],[batch_size,1000],[batch_size,1000],...5次]
        batch_size = 1
        image_features = [image_features[i*batch_size : i*batch_size + batch_size] for i in range(image_num)]
        #  print(len(image_features))
        # 遍历batch

        for batch_clip in range(1):
            I0 = image_features[0][batch_clip].unsqueeze(0)
            I1 = image_features[1][batch_clip].unsqueeze(0)
            I2 = image_features[2][batch_clip].unsqueeze(0)
            I3 = image_features[3][batch_clip].unsqueeze(0)
            I4 = image_features[4][batch_clip].unsqueeze(0)
            # I.shape is [1, 5, 1000]
            I = torch.stack([I0, I1, I2, I3, I4], dim=1)

            # size is [1,2] because model_gcn has tp_layer layer.
            output_features = model_gcn(I)
            # print(output_features.shape)

            loss = criterion(output_features, image_label.to(device))

            # 进行更新
            optimizer_gcn.zero_grad()
            optimizer_tl.zero_grad()
            optimizer_backbone.zero_grad()

            loss.backward()

            optimizer_tl.step()
            optimizer_gcn.step()
            optimizer_backbone.step()

            if batchidx % 100 == 0:
                print(epoch, batchidx, loss.cpu().item())
                viz.line([loss.cpu().item()], [epoch * len(train_dataloader) + batchidx], win='loss', update='append')

            if batchidx < 180:
                viz.line([output_features.argmax(dim=1).item()], [batchidx], win='predict', update='append')

            # if batchidx >= 302:
            #     return  loss

    return  loss

        #  # save checkpoint
        #  n = len(loader) * epoch + i
        #  if n % args.checkpoints == 0:
        #      path = os.path.join(
        #          args.exp,
        #          'checkpoints',
        #          'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
        #      )
        #      if args.verbose:
        #          print('Save checkpoint at: {0}'.format(path))
        #      torch.save({
        #          'epoch': epoch + 1,
        #          'arch': args.arch,
        #          'state_dict': model.state_dict(),
        #          'optimizer' : opt.state_dict()
        #      }, path)

        # target = target.to(device)
        # input_var = torch.autograd.Variable(input_tensor.to(device))
        # target_var = torch.autograd.Variable(target)
        #
        # output = model(input_var)
        # loss = criterion(output, target_var)
        #
        # # record loss
        # losses.update(loss.data, input_tensor.size(0))
        #
        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # optimizer_tl.zero_grad()
        # loss.backward()
        # optimizer.step()
        # optimizer_tl.step()

    ###########################################################################3
    # 对模型进行保存


if __name__ == "__main__":
    main()

