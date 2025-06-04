import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import clip
from utils import spectral_norm as SpectralNorm

class LabelNet(nn.Module):
    def __init__(self, bit, num_classes):
        super().__init__()

        self.curr_dim = 16
        self.size = 32
        self.feature = nn.Sequential(nn.Linear(num_classes, 4096), nn.ReLU(True), nn.Linear(4096, self.curr_dim * self.size * self.size))
        conv2d = [
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.InstanceNorm2d(64),
            nn.Tanh()
        ]
        self.conv2d = nn.Sequential(*conv2d)

    def forward(self, label_feature):
        label_feature = self.feature(label_feature)
        label_feature = label_feature.view(label_feature.size(0), self.curr_dim, self.size,self.size)
        label_feature = self.conv2d(label_feature)
        return label_feature


class TextNet(nn.Module):
    def __init__(self, dim_text, bit, num_classes):
        super(TextNet, self).__init__()
        self.curr_dim = 16
        self.size = 32
        self.feature = nn.Sequential(nn.Linear(dim_text, 4096), nn.ReLU(True), nn.Linear(4096,
                                                                                         self.curr_dim * self.size * self.size))
        conv2d = [
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.InstanceNorm2d(64),
            nn.Tanh()
        ]
        self.conv2d = nn.Sequential(*conv2d)

    def forward(self, text_feature):
        text_feature = self.feature(text_feature)
        text_feature = text_feature.view(text_feature.size(0), self.curr_dim, self.size,self.size)
        text_feature = self.conv2d(text_feature)
        return text_feature


# Progressive Fusion Module
class PrototypeNet(nn.Module):
    def __init__(self, dim_text, bit, num_classes, channels=64, r=4):
        super().__init__()
        self.labelnet = LabelNet(bit, num_classes)
        self.textnet = TextNet(dim_text, bit, num_classes)
        inter_channels = int(channels // r)  # 16
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()


        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.Tanh()
        )
        self.hashing = nn.Sequential(nn.Linear(4096, bit), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(4096, num_classes), nn.Sigmoid())

    def forward(self, label_feature, text_feature):
        label_feature = self.labelnet(label_feature)
        text_feature = self.textnet(text_feature)
        xa = label_feature + text_feature

        xl1 = self.local_att(xa)
        xg1 = self.global_att(xa)
        xlg1 = xl1 + xg1
        wei1 = self.sigmoid(xlg1)
        xi1 = label_feature * wei1 + text_feature * (1 - wei1)

        xl2 = self.local_att2(xi1)
        xg2 = self.global_att(xi1)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        mixed_feature = label_feature * wei2 + text_feature * (1 - wei2)

        mixed_tensor = self.conv2d(mixed_feature)
        mixed_tensor = mixed_tensor.view(mixed_tensor.size(0), -1)
        mixed_hashcode = self.hashing(mixed_tensor)
        mixed_label = self.classifier(mixed_tensor)
        return mixed_feature, mixed_hashcode, mixed_label


class SemanticTranslatorter(nn.Module):
    def __init__(self,args):
        super(SemanticTranslatorter, self).__init__()
        self.args=args
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(64, 48, kernel_size=3, stride=1, padding=2, bias=False),#14
            nn.InstanceNorm2d(48, affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 12, kernel_size=4, stride=2, padding=1, bias=False),#28
            nn.InstanceNorm2d(12, affine=False),
            nn.ReLU(inplace=True)
           )
        '''
        self.transform1 = nn.Sequential(
            nn.ConvTranspose2d(12, 1, kernel_size=6, stride=6, padding=0, bias=False),  # 168
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
        )
        self.transform2 = nn.Sequential(
            nn.ConvTranspose2d(12, 1, kernel_size=5, stride=3, padding=1, bias=False),  # 84
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
        )
        '''
        '''
        self.transform1 = nn.Sequential(
            nn.ConvTranspose2d(12, 1, kernel_size=4, stride=2, padding=1, bias=False),  # 56
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
        )
        self.transform2 = nn.Sequential(
            nn.ConvTranspose2d(12, 1, kernel_size=3, stride=1, padding=1, bias=False),  # 28
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
        )
        '''
        '''
        self.transform1 = nn.Sequential(
            nn.ConvTranspose2d(12, 1, kernel_size=3, stride=3, padding=3, bias=False),  # want:75,real:78
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
        )
        self.transform2 = nn.Sequential(
            nn.ConvTranspose2d(12, 1, kernel_size=7, stride=1, padding=0, bias=False),  # want:37,real:34
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
        )
        '''

        self.transform1=nn.Sequential(
            nn.ConvTranspose2d(12, 1, kernel_size=6, stride=4, padding=1, bias=False),#112
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
            )
        self.transform2 = nn.Sequential(
            nn.ConvTranspose2d(12, 1, kernel_size=4, stride=2, padding=1, bias=False),#56
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
            )

    def expand_matrix(self,input_matrix,size):
        new_size=(input_matrix.size(0),input_matrix.size(1),size,size)
        original_size = input_matrix.size()

        expanded_matrix = torch.zeros(new_size)

        start_x = torch.randint(0, new_size[2] - original_size[2] + 1, (1,)).item()
        start_y = torch.randint(0, new_size[3] - original_size[3] + 1, (1,)).item()

        expanded_matrix[:,:,start_x:start_x + original_size[2], start_y:start_y + original_size[3]] = input_matrix
        return expanded_matrix.cuda()
    def forward(self, label_feature):
        label_feature = self.transform(label_feature)
        be_label_feature1 = self.transform1(label_feature)
        be_label_feature2 = self.transform2(label_feature)
        label_feature1=self.expand_matrix(be_label_feature1,224)
        label_feature2 = self.expand_matrix(be_label_feature2, 112)
        return label_feature1,label_feature2


class Generatorter(nn.Module):
    def __init__(self,args):
        super(Generatorter, self).__init__()
        self.translator = SemanticTranslatorter(args)
        # Image Encoder
        curr_dim = 64
        self.preprocess = nn.Sequential(
            nn.Conv2d(6, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.firstconv = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        self.preprocess2=nn.Sequential(
            nn.Conv2d(1+curr_dim * 2, curr_dim * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        self.secondconv = nn.Sequential(
            nn.Conv2d(curr_dim * 2, curr_dim * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 4),
            nn.ReLU(inplace=True))
        # Residual Block
        self.residualblock = nn.Sequential(
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'))
        # Image Decoder
        self.firstconvtrans = nn.Sequential(
            nn.ConvTranspose2d(curr_dim * 4, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        self.preprocess3 = nn.Sequential(
            nn.Conv2d(curr_dim * 4, curr_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        self.secondconvtrans = nn.Sequential(
            nn.ConvTranspose2d(curr_dim * 2, curr_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.process = nn.Sequential(
            nn.Conv2d(curr_dim * 2, curr_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.residual = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh())
    def forward(self, x, mixed_feature):
        mixed_feature1,mixed_feature2  = self.translator(mixed_feature)
        tmp_tensor = torch.cat((x[:,0,:,:].unsqueeze(1), mixed_feature1, x[:,1,:,:].unsqueeze(1), mixed_feature1, x[:,2,:,:].unsqueeze(1), mixed_feature1), dim = 1)
        tmp_tensor = self.preprocess(tmp_tensor)
        tmp_tensor_first = tmp_tensor
        tmp_tensor = self.firstconv(tmp_tensor)
        tmp_tensor=torch.cat((tmp_tensor,mixed_feature2),dim = 1)
        tmp_tensor=self.preprocess2(tmp_tensor)
        tmp_tensor_second = tmp_tensor
        tmp_tensor = self.secondconv(tmp_tensor)
        tmp_tensor = self.residualblock(tmp_tensor)
        tmp_tensor = self.firstconvtrans(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_second, tmp_tensor), dim = 1)
        tmp_tensor=self.preprocess3(tmp_tensor)
        tmp_tensor = self.secondconvtrans(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_first, tmp_tensor), dim = 1)
        tmp_tensor = self.process(tmp_tensor)
        tmp_tensor = torch.cat((x, tmp_tensor), dim = 1)
        tmp_tensor = self.residual(tmp_tensor)
        return tmp_tensor

# ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine))
    def forward(self, x):
        return x + self.main(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes, image_size=224, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / (2**repeat_num))
        self.main = nn.Sequential(*layers)
        self.fc = nn.Conv2d(curr_dim, num_classes + 1, kernel_size=kernel_size, bias=False)


    def forward(self, x):
        h = self.main(x)
        out = self.fc(h)
        return out.squeeze()


# GAN Objectives
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)


    def get_target_tensor(self, label, target_is_real):
        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, real_label], dim=-1)
        else:
            fake_label = self.fake_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, fake_label], dim=-1)
        return target_tensor
    
    def __call__(self, prediction, label, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(label, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

# Learning Rate Scheduler
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.n_epochs,
                                                   eta_min=0)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



