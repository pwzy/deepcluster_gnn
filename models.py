import math

import numpy as np
import torch
import torch.nn as nn
 
# (number of filters, kernel size, stride, pad)
CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}

class AlexNet(nn.Module):
    def __init__(self, features, num_classes):
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


    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

# 进行类的测试
if __name__ == "__main__":
    model = AlexNet()


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



