import torch
from torch import nn
from .basic_module import BasicModule
from torchvision import models
import clip
vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}
class VGG(BasicModule):
    def __init__(self, model_name, bit):
        super(VGG, self).__init__()
        self.module_name = "image_model"
        original_model = vgg_dict[model_name](pretrained=True)
        self.features = original_model.features
        self.cl1 = nn.Linear(25088, 4096)
        self.cl1.weight = original_model.classifier[0].weight
        self.cl1.bias = original_model.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[3].weight
        cl2.bias = original_model.classifier[3].bias

        self.classifier = nn.Sequential(
            self.cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = x / 255.
        x = (x-self.mean)/self.std
        f = self.features(x)
        f = f.contiguous().view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}
class ResNet(BasicModule):
    def __init__(self, model_name, hash_bit):
        super(ResNet, self).__init__()
        self.module_name = "image_model"
        model_resnet = resnet_dict[model_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                             self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

        self.activation = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = x / 255.
        x = (x-self.mean)/self.std
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        y = self.activation(alpha*y)
        return y

class AlexNet(BasicModule):
    def __init__(self, bit):
        super(AlexNet, self).__init__()
        self.module_name = "image_model"
        original_model = models.alexnet(pretrained=True)
        self.features = original_model.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = original_model.classifier[1].weight
        cl1.bias = original_model.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[4].weight
        cl2.bias = original_model.classifier[4].bias

        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = x / 255.
        x = (x-self.mean)/self.std
        f = self.features(x)
        f = f.contiguous().view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y

class CLIPHashNet(BasicModule):
    def __init__(self, backbone,hash_bit):
        super(CLIPHashNet, self).__init__()
        # 加载预训练的CLIP模型
        self.module_name = "image_model"
        if 'CNN' in backbone:
            self.clip_model, _ = clip.load("RN50", device='cuda', jit=False)  # 确保模型加载到正确的设备上
        else:
            self.clip_model, _ = clip.load("ViT-B/32", device='cuda', jit=False)  # 确保模型加载到正确的设备上
        for param in self.clip_model.parameters():
            param.requires_grad = False  # 冻结CLIP模型的所有参数
        # 定义一个新的哈希层
        self.hash_layer = nn.Linear(self.clip_model.visual.output_dim, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        # 定义激活函数
        self.activation = nn.Tanh()
        # 设置归一化所需的均值和标准差（CLIP的默认值）
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).cuda()

    def forward(self, x, alpha=1):
        # 对输入图像进行归一化处理
        x = x / 255.0
        x = (x - self.mean) / self.std
        # 通过CLIP的视觉部分提取特征
        x = self.clip_model.encode_image(x)
        x = x.float()
        # 应用哈希层
        y = self.hash_layer(x)
        # 应用激活函数
        y = self.activation(alpha * y)
        return y

class ImgModule(BasicModule):
    def __init__(self, bit, pretrain_model=None):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        # fc8
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self.classifier.weight.data = torch.randn(bit, 4096) * 0.01
        self.classifier.bias.data = torch.randn(bit) * 0.01
        self.mean = torch.zeros(3, 224, 224)
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        if x.is_cuda:
            x = x - self.mean.cuda()
        else:
            x = x - self.mean
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        return x


