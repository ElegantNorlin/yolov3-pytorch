from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53
# filter_in输入特征通道数
# filter_out输出特征通道数
# kernel_size卷积核大小
def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
# filters_list:[,]
# out_filter = 75 最后输出特征的通道数
# in_filters = out_filters[]列表中定义个三个特征层的通道数
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,stride=1, padding=0, bias=True)
    ])
    return m

class YoloBody(nn.Module):
    def __init__(self, anchor, num_classes):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型,保存在self.backbone变量中
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53(None)

        # out_filters : [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        # final_out_filter0 = 75 = 3 * (4 + 1 + 20)
        # 3个先验框
        # 4个先验框调整参数
        # 1：判断先验框的内部是否有物体
        # 20(num_classes)中物体，voc数据集是20中物体
        final_out_filter0 = len(anchor[0]) * (5 + num_classes)
        # out_filters[-1] = 1024
        # final_out_filter0 = 75
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)

        final_out_filter1 = len(anchor[1]) * (5 + num_classes)
        # 1x1卷积调整通道数
        self.last_layer1_conv = conv2d(512, 256, 1)
        # 上采样(分辨率变高)
        # scale_factor=2 为输出为输入的多少倍
        # mode='nearest' 为上采样的算法
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 现在得到一个26x26x256的特征层
        # 特征合并在前向传播中定义
        # 接下来按照网络架构图应该为5个卷积和两次卷积
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        final_out_filter2 = len(anchor[2]) * (5 + num_classes) 
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 现在得到一个52x52x128的特征层
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)

    # 前向传播过程
    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        # x0 : 13,13,1024
        # x1 : 26,26,512
        # x2 : 52,52,256
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 五次卷积过程
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        # out0为最后两次卷积后的结果
        # out0_branch为5次卷积后的特征层
        out0, out0_branch = _branch(self.last_layer0, x0)

        # 卷积 + 上采样过程
        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 特征层堆叠
        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        # 
        out2, _ = _branch(self.last_layer2, x2_in)
        return out0, out1, out2

