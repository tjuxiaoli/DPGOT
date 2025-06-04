import torch
import torch.nn as nn
import scipy.io as scio


class Attacked_Model(nn.Module):
    def __init__(self, method, dataset, bit, attacked_models_path, dataset_path,backbone):
        super(Attacked_Model, self).__init__()
        self.method = method
        self.dataset = dataset
        self.bit = bit
        self.backbone = backbone

        if self.dataset == 'FLICKR':
            tag_dim = 1386
            num_label = 24
        if self.dataset == 'COCO':
            tag_dim = 1024
            num_label = 80



        load_img_path = attacked_models_path + str(self.method) + '_' + self.backbone + '_' + str(self.bit) + '/image_model.pth'
        load_txt_path = attacked_models_path + str(self.method) + '_' + self.backbone + '_' + str(self.bit) + '/text_model.pth'
        if self.backbone=='AlexNet':
            from attacked_methods.DCMH.img_module import AlexNet
            from attacked_methods.DCMH.txt_module import TxtModule
            self.image_hashing_model = AlexNet(self.bit)
            self.text_hashing_model = TxtModule(tag_dim, self.bit)
        elif 'ResNet' in self.backbone:
            from attacked_methods.DCMH.img_module import ResNet
            from attacked_methods.DCMH.txt_module import TxtModule
            self.image_hashing_model = ResNet(self.backbone,self.bit)
            self.text_hashing_model = TxtModule(tag_dim, self.bit)
        elif 'CLIP' in self.backbone:
            from attacked_methods.DCMH.img_module import CLIPHashNet
            from attacked_methods.DCMH.txt_module import TxtModule
            self.image_hashing_model = CLIPHashNet(self.backbone, self.bit)
            self.text_hashing_model = TxtModule(tag_dim, self.bit)
        else :
            from attacked_methods.DCMH.img_module import ImgModule
            from attacked_methods.DCMH.txt_module import TxtModule
            self.image_hashing_model = ImgModule(self.bit)
            self.text_hashing_model = TxtModule(tag_dim, self.bit)

        self.image_hashing_model.load(load_img_path)
        self.text_hashing_model.load(load_txt_path)
        self.image_hashing_model.cuda().eval()
        self.text_hashing_model.cuda().eval()

    def generate_image_feature(self, data_images):
        num_data = data_images.size(0)
        B = torch.zeros(num_data, self.bit)
        for i in range(num_data):
            output = self.image_hashing_model(data_images[i].type(torch.float).unsqueeze(0).cuda())
            B[i, :] = output.data
        return B
    
    def generate_text_feature(self, data_texts):
        num_data = data_texts.size(0)
        B = torch.zeros(num_data, self.bit)
        for i in range(num_data):
            output = self.text_hashing_model(data_texts[i].type(torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(3).cuda())
            B[i, :] = output.data
        return B

    def generate_image_hashcode(self, data_images):
        num_data = data_images.size(0)
        B = torch.zeros(num_data, self.bit)
        for i in range(num_data):
            output = self.image_hashing_model(data_images[i].type(torch.float).unsqueeze(0).cuda())
            B[i, :] = output.data
        return torch.sign(B)

    def generate_text_hashcode(self, data_texts):
        num_data = data_texts.size(0)
        B = torch.zeros(num_data, self.bit)
        for i in range(num_data):
            output = self.text_hashing_model(data_texts[i].type(torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(3).cuda())
            B[i, :] = output.data
        return torch.sign(B)

    def image_model(self, data_images):
        output = self.image_hashing_model(data_images)
        return output
    def text_model(self, data_texts):
        output = self.text_hashing_model(data_texts.type(torch.float).unsqueeze(1).unsqueeze(3).cuda())
        return output