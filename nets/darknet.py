import math
from collections import OrderedDict

import torch
import torch.nn as nn

'''
darknet.py文件为用pytorch搭建的darknet53特征提取网络
'''

#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#

# 定义残差块网络结构，需要堆叠残差块的时候直接循环此残差块网络即可
class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        '''
        首先用1*1的卷积下降通道数
        再用3*3的卷积扩张通道数
        这样可以减少参数量，加速特征提取的速度
        '''
        # inplanes为输入通道数
        # planes[0]为输出通道数
        # kernel_size=1为卷积核尺寸
        # stride=1为卷积步长
        # padding=0填充厚度
        # bias=False这里没有用到偏置项所以值为False
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)
    # 残差网络前向传播
    def forward(self, x):
        # 定义残差边
        residual = x

        # 残差主干
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # 残差块的结构：残差边与主干合并
        out += residual
        return out

# 定义了Darknet网络类
class DarkNet(nn.Module):
    # 参数layers为列表，列表的每一个参数为残差块的堆叠次数
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        # 定义第一个卷积模块中的32通道
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        # 卷积操作得到416,416,32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # 标准化
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # LeakyReLU激活函数
        self.relu1 = nn.LeakyReLU(0.1)

        # 开始堆叠残差块
        # 堆叠第一个大残差块
        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 堆叠第二个大残差块
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 堆叠第三个大残差块
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 堆叠第四个大残差块
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 堆叠第五个大残差块
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    # 定义DarkNet网络中的每个堆叠的大残差块
    # planes为列表，[输入通道数,输出通道数]
    # blocks为数值，当前大残差块堆叠了多少次残差块
    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样(可以理解为图片的分辨率下降，成为“下采样”或“降采样”)，步长为2，卷积核大小为3
        # 卷积层
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        # 定义标准化层
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        # 定义LeakyReLU激活函数
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        # blocks是定义了该残差块的使用(堆叠)的次数,也就是我们传入的那个列表中的数值。
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        # OrderedDict()按照元素插入的顺序输出元素
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # 这里对最后三个特征层单独拿出来命名，因为后面的网络需要用到这三个特征层
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    # 使用DarkNet类生成DarkNet对象命名为model
    # 传入的列表对应了相应的残差块使用的次数
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
