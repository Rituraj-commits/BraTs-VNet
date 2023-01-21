"""Implementation based on 
https://github.com/black0017/MedicalZooPytorch/blob/8f40dab689841d7ff0e36aa5e583a1a1509fac3d/lib/medzoo/Densenet3D.py#L4"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class _HyperDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_channels, drop_rate):
        super(_HyperDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features,
                                           num_output_channels, kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_HyperDenseLayer, self).forward(x)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)

        return torch.cat([x, new_features], 1)


class _HyperDenseBlock(nn.Sequential):
    """
    Constructs a series of dense-layers based on in and out kernels list
    """

    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlock, self).__init__()
        out_kernels = [1, 25, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 9

        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])

        #print("out:", out_kernels)
        #print("in:", in_kernels)

        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _HyperDenseBlockEarlyFusion(nn.Sequential):
    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlockEarlyFusion, self).__init__()
        out_kernels = [1, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 8

        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])

        #print("out:", out_kernels)
        #print("in:", in_kernels)

        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class SinglePathDenseNet(nn.Sequential):
    def __init__(self, in_channels, classes=4, drop_rate=0.1, return_logits=True, early_fusion=False):
        super(SinglePathDenseNet, self).__init__()
        self.return_logits = return_logits
        self.features = nn.Sequential()
        self.num_classes = classes
        self.input_channels = in_channels

        if early_fusion:
            block = _HyperDenseBlockEarlyFusion(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 52:
                total_conv_channels = 477
            else:
                if in_channels == 3:
                    total_conv_channels = 426
                else:
                    total_conv_channels = 503

        else:
            block = _HyperDenseBlock(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 2:
                total_conv_channels = 452
            else:
                total_conv_channels = 454

        self.features.add_module('denseblock1', block)

        self.features.add_module('conv1x1_1', nn.Conv3d(total_conv_channels,
                                                        400, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_1', nn.Dropout(p=0.5))

        self.features.add_module('conv1x1_2', nn.Conv3d(400,
                                                        200, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_2', nn.Dropout(p=0.5))

        self.features.add_module('conv1x1_3', nn.Conv3d(200,
                                                        150, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_3', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(150,
                                                           self.num_classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))

    def forward(self, x):
        features = self.features(x)
        
        if self.return_logits:
            out = self.classifier(features)
            return out

        else:
            return features
