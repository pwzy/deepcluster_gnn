# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import numpy as np
import torch
import torch.nn as nn

  
__all__ = [ 'AlexNet', 'alexnet']
 
# (number of filters, kernel size, stride, pad)
CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}

class GCN_Module(nn.Module):
    def __init__(self):
        super(GCN_Module, self).__init__()

        # self.cfg = cfg

        # NFR = cfg.num_features_relation  # 256
        # 图的特征的个数
        NFR = 256  # 256

        # NG = cfg.num_graph  # 4
        # N = cfg.num_boxes  # 13
        # T = cfg.num_frames  # 10
        #  NG, N, T = 4, 13, 10 # 分别为新建的图的个数，box的数量，帧的数量
        NG = 4

        # NFG = cfg.num_features_gcn  # 1024
        # NFG_ONE = NFG  # 1024
        # 为初始的box的特征数
        NFG = 1000
        NFG_ONE = 1000

        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])

        self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])

    def forward(self, graph_boxes_features):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        # B, N, NFG = graph_boxes_features.shape  # 1 15 1024
        B, N, NFG = 1,5,1000
        # NFR = self.cfg.num_features_relation  # 256
        NFR = 256
        # NG = self.cfg.num_graph  # 4
        NG = 4
        NFG_ONE = NFG  # 1024

        # OH, OW = self.cfg.out_size  # 57, 87
        #  OH, OW = 57,87
        # pos_threshold = self.cfg.pos_threshold  # 0.2
        #  pos_threshold = 0.2

        # Prepare position mask
        #  graph_boxes_positions = boxes_in_flat  # B*T*N, 4  [15,4]
        #  graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        #  graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        #  graph_boxes_positions = graph_boxes_positions[:, :2].reshape(B, N, 2)  # B*T, N, 2  [1, 15 ,2]

        #  graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions,
        #                                                    graph_boxes_positions)  # B, N, N  [1, 15 ,15]
        #
        #  position_mask = (graph_boxes_distances > (pos_threshold * OW))  # [1, 15 ,15]  is bool value

        relation_graph = None
        graph_boxes_features_list = []
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # B,N,NFR  ([1, 15, 256])
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # B,N,NFR  ([1, 15, 256])

            #             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
            #             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(1, 2))  # B,N,N  ([1, 15, 15])

            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*N*N, 1  ([225, 1])

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B, N, N)  # ([1, 15, 15])
            # 关闭position_mask
            #  relation_graph[position_mask] = -float('inf')

            relation_graph = torch.softmax(relation_graph, dim=2)  # ([1, 15, 15])

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](
                torch.matmul(relation_graph, graph_boxes_features))  # B, N, NFG_ONE  ([1, 15, 1024])
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)

            # 去掉激活函数层
            #  one_graph_boxes_features = F.relu(one_graph_boxes_features)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.sum(torch.stack(graph_boxes_features_list),
                                         dim=0)  # B, N, NFG  # ([1, 15, 1024]) 合并4个[1, 15, 1024]

        return graph_boxes_features, relation_graph


class MY_Net(nn.Module):
    def __init__(self, num_classes):
        super(MY_Net, self).__init__()
        # 创建不带激活层的网络
        self.gcn = GCN_Module()
        self.fc = nn.Sequential(
            nn.Linear(1000 * 5, 100 * 5),
            nn.ReLU(),
            nn.Linear(100 * 5, 100 * 5),
            nn.ReLU(),
            nn.Linear(100 * 5, 100 * 5),
        )
        # 创建带有激活层的top_layer
        self.top_layer = nn.Sequential(
                # 由于只有一个batch，调用正则层会报错
                #  nn.BatchNorm1d(1000 * 5),
                #  nn.ReLU(),
                #  nn.Linear(100*5,100*5),
                nn.ReLU(),
                nn.Linear(100*5,10*5),
                nn.ReLU(),
                nn.Linear(10*5, num_classes),
                nn.Sigmoid(),
                )

    def forward(self, x):
        # x代表传来的提取的图像的特征 x:[1,5,1000] => [1,5,1000]
        x, _ = self.gcn(x)

        x = x.view(1, 5*1000)

        x = self.fc(x)
        
        if self.top_layer:
            x = self.top_layer(x)
        return x

if __name__ == "__main__":

    
    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MY_Net(num_classes=2).to(device)

    boxes_features = torch.randn(1,5,1000).to(device)
    output = model(boxes_features)
    # 输出为 [1,2]
    print(output.shape)
    #  print(output.shape)

    
    



class AlexNet(nn.Module):
    def __init__(self, features, num_classes, sobel):
        super(AlexNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(256 * 6 * 6, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True))

        self.top_layer = nn.Linear(4096, num_classes)
        self._initialize_weights()

        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def alexnet(sobel=False, bn=True, out=1000):
    dim = 2 + int(not sobel)
    model = AlexNet(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel)
    return model
