import random
import os
import torch
import argparse
import numpy as np

from load_data import load_dataset, split_dataset, allocate_dataset, Dataset_Config
from attack_model import DPGOT

parser = argparse.ArgumentParser()
# dataset setting
parser.add_argument('--dataset', dest='dataset', default='FLICKR', choices=['WIKI', 'IAPR', 'FLICKR', 'COCO', 'NUS'], help='name of the dataset')
# attacked model setting

parser.add_argument('--attacked_method', dest='attacked_method', default='DCMH', choices=['DCMH', 'CPAH', 'DADH', 'DJSRH', 'JDSH', 'DGCPN'], help='deep cross-modal hashing methods')
parser.add_argument('--backbone',dest='backbone',default='ResNet50',choices=['AlexNet','ResNet50','ResNet34','CNN-F','CLIP','CLIPCNN'])
parser.add_argument('--output_dir', dest='output_dir', default='DCMH_ResNet50_32', help='the name of output')
###################################################################################################################
parser.add_argument('--bit', dest='bit', type=int, default=32, choices=[16, 32, 64, 128], help='length of the hashing code')
parser.add_argument('--transfer_backbone',dest='transfer_backbone',default='ResNet34',choices=['ResNet50','ResNet34','CNN-F','CLIP','CLIPCNN'])
parser.add_argument('--transfer_attack', dest='transfer_attack',default=True, action='store_true', help='transfer_attack')
parser.add_argument('--transfer_attacked_method', dest='transfer_attacked_method', default='DCMH', choices=['DCMH', 'CPAH', 'DADH', 'DJSRH', 'JDSH', 'DGCPN'])
parser.add_argument('--transfer_bit', dest='transfer_bit', type=int, default=32, choices=[16, 32, 64, 128], help='length of the hashing code')

parser.add_argument('--dataset_path', dest='dataset_path', default='./Datasets/', help='path of the dataset')
parser.add_argument('--attacked_models_path', dest='attacked_models_path', default='./attacked_models/', help='path of attacked models')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', choices=['0', '1', '2', '3'])
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=50, help='number of epoch')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=10, help='print the debug information every print_freq iterations')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--sample_dir', dest='sample', default='samples/', help='output image are saved here during training')
parser.add_argument('--output_path', dest='output_path', default='./outputs/', help='models are saved here')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

Dcfg = Dataset_Config(args.dataset, args.dataset_path)

X, Y, L = load_dataset(Dcfg.data_path)
X, Y, L = split_dataset(X, Y, L, Dcfg.query_size, Dcfg.training_size, Dcfg.database_size)
Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L = allocate_dataset(X, Y, L)


model = DPGOT(args=args, Dcfg=Dcfg)

model.test_attacked_model(Te_I, Te_T, Te_L, Db_I, Db_T, Db_L)

model.train_prototypenet(Tr_I, Tr_T, Tr_L)
model.test_prototypenet(Te_I,Te_T, Te_L, Db_I, Db_T, Db_L,Tr_T,args)

model.trainiao(Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_T, Te_L,args)
model.test(Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)#

if args.transfer_attack:
    model.transfer_attack(Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)