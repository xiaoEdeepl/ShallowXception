import pretrainedmodels
import torch.nn.functional as F
import torch.nn as nn
# from keras.src.applications.xception import Xception
from pretrainedmodels.models.xception import Block, SeparableConv2d
from pyexpat import features


# 定义一个继承自xception的自定义模型
class Xception(nn.Module):
    def __init__(self, classes=2):
        super(Xception, self).__init__()
        # 加载xception模型，使用预训练权重
        self.classes = classes
        self.model = pretrainedmodels.__dict__['xception'](pretrained=False)

        # 替换最后的分类头，适应2类分类任务
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, classes)

    def forward(self, x):
        return self.model(x)

    def name(self):
        return "Xception"

    def extract_features(self, x):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        return x



class ShallowXception(nn.Module):
    def __init__(self, num_classes=2):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(ShallowXception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        # self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        # self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        # self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        #
        # self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        # self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        # self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        # self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------
    # 提取全局平均池化前的特征，用于t-SNE可视化
    def extract_features(self, input):
        features = self.features(input)
        features = self.relu(features)
        return features

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.block8(x)
        # x = self.block9(x)
        # x = self.block10(x)
        # x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    def name(self):
        return "ShallowXception"


if __name__ == '__main__':
    model = ShallowXception()
    print(model)