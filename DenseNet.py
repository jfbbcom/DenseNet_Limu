import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 创建一个卷积核块
# input_channels：输入特征图的通道数。
# num_channels：输出特征图的通道数。
def conv_block(input_channels, num_channels):
    # 创建序列化的模型对象。该对象按照顺序包含了以下几个层：
    # nn.BatchNorm2d(input_channels)：对输入通道进行批标准化操作，用于规范化输入数据。
    # nn.ReLU()：应用ReLU激活函数，增加非线性特征表达能力。
    # nn.Conv2d()：使用3x3的卷积核对输入特征图进行卷积操作，输出特征图的通道数为num_channels，
    # 并使用padding为1保持输入输出尺寸相同。
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    # num_convs：密集连接块中卷积层的数量
    # input_channels：输入特征图的通道数
    # num_channels：增长率,每一层的额外通道数
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        # 储存num_convs个卷积块
        layer = []
        for i in range(num_convs):
            # 每个conv_block输入通道不一样，输出通道一样
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        # *layer表示将layer列表解包，将其中的卷积块作为参数传递给nn.Sequential函数。
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 通过循环，连接通道维度上每个块的输⼊和输出
            X = torch.cat((X, Y), dim=1)
        return X

# [have a try]
# blk = DenseBlock(2, 3, 10)
# X = torch.randn(4, 3, 8, 8)
# Y = blk(X)
# Y.shape

# 7.7.3过渡层
def transition_block(input_channels, num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

# # [have a try]
# blk = transition_block(23, 10)
# blk(Y).shape

# 7.7.4 DenseNet模型
# 我们来构造DenseNet模型。DenseNet⾸先使⽤同ResNet⼀样的单卷积层和最⼤汇聚层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# num_channels为当前的通道数，growth_rate：增长率
num_channels, growth_rate = 64, 32
# 在DenseNet中每一块denseblock的卷积层数量均为4
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
# 给blks列表加入赋值好的DenseBlock块与transition_block，
# DenseBlock输入为64通道，增长率为32通道，每个卷积层有4层
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上⼀个稠密块的输出通道数
    # num_channels = num_channels + num_convs * growth_rate
    num_channels += num_convs * growth_rate
    # 如果当前迭代的 DenseBlock 不是最后一个 DenseBlock。添加转换层（transition block），用于减少通道数
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    #全连接层，将展平后的特征向量映射到长度为 10 的输出向量，用于分类任务。
    nn.Linear(num_channels, 10))

# 学习率、学习轮数、批样本大小
lr, num_epochs, batch_size = 0.001, 10, 256
# 训练迭代器 测试迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# 训练
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()
