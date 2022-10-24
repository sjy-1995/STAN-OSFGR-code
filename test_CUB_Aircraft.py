# coding:utf-8
import os
import argparse
import datetime
import time
import pandas as pd
import importlib

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn

from methods.ARPL.arpl_models import gan
from methods.ARPL.arpl_models.arpl_models import classifier32ABN
from methods.ARPL.arpl_models.wrapper_classes import TimmResNetWrapper
from methods.ARPL.arpl_utils import save_networks
from methods.ARPL.core import train, train_cs, test

from utils.utils import init_experiment, seed_torch, str2bool
from utils.schedulers import get_scheduler
from data.open_set_datasets import get_class_splits, get_datasets
from models.model_utils import get_model

from config import exp_root

import timm
from methods.ARPL.arpl_utils import AverageMeter
from tqdm import tqdm
import numpy as np
from methods.ARPL.core import evaluation
import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score

import pickle
from test.utils import closed_set_acc, acc_at_95_tpr, compute_auroc, compute_aupr, compute_oscr
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score

import scipy.io as sio
import math

# swin transformer as the backbone
from swin_transformer import SwinTransformer   # the more complex file

from openSetClassifier_MoEP_AE_for_fine_grained import openSetClassifier


parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cub', help="")
parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
# parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=448)
# parser.add_argument('--image_size', type=int, default=224)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--lr', type=float, default=1, help="learning rate for model")
# parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
# parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--max-epoch', type=int, default=600)
# parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts')
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
# parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing."
parser.add_argument('--label_smoothing', type=float, default=0.3, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
# parser.add_argument('--model', type=str, default='classifier32')
parser.add_argument('--model', type=str, default='timm_resnet50_pretrained')
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
# parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_m', type=int, default=30)
# parser.add_argument('--rand_aug_n', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=2)

# misc
# parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--checkpt_freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--train_feat_extractor', default=True, type=str2bool,
                        help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')

parser.add_argument('--osr_mode', default='max_softmax', type=str, help='{entropy, max_softmax}')

# parser.add_argument('--exp_id', type=str, default='(17.02.2022_|_23.656)')   # cub exp1 448 * 448


# ###################################### self-defined model ######################################
mask_threshold = 0.5


class mytry5_20220512_v3_1(nn.Module):  # for ViT-B/16
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220512_v3_1, self).__init__()
        self.swinB = transformer
        # self.layernorm = nn.LayerNorm(768)
        # self.fc = nn.Linear(768, num_classes)
        self.fc = nn.Linear(1024, num_classes)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()

        # self.Upsample448 = nn.UpsamplingBilinear2d(size=(448, 448))
        # self.Upsample112 = nn.UpsamplingBilinear2d(size=(112, 112))
        # self.Upsample56 = nn.UpsamplingBilinear2d(size=(56, 56))
        # self.Upsample28 = nn.UpsamplingBilinear2d(size=(28, 28))
        # self.Upsample14 = nn.UpsamplingBilinear2d(size=(14, 14))
        # self.Upsample1 = nn.Upsample(size=(2048), mode='bilinear')

        # self.layer1_conv1_1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, bias=False)
        # self.layer2_conv1_1 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, bias=False)
        # self.layer3_conv1_1 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, bias=False)
        # self.layer4_conv1_1 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, bias=False)

        self.attn_fc1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.attn_fc2 = nn.Linear(128, 1024)

        # self.LSTMcell = LSTMCell_mytry5_20220512_v2(num_classes=num_classes)

        self.num_classes = num_classes

    def forward(self, x, need_feature=False):
        x = self.swinB.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.swinB.pos_drop(x)

        num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            # print(x.shape)   # (b, 3136=56*56, 256), (b, 784=28*28, 512), (b, 196=14*14, 1024), (b, 196=14*14, 1024)
            # all_features += x.unsqueeze(0)
            x_ = x.permute(0, 2, 1)
            feature_maps = x_.view(x_.shape[0], x_.shape[1], int(math.sqrt(x_.shape[2])), int(math.sqrt(x_.shape[2]))) # (b, 256, 56, 56)
            # if num == 0:
            #     attention_map = self.layer1_conv1_1(feature_maps)   # (b, 1, 56, 56)
            # elif num == 1:
            #     attention_map = self.layer2_conv1_1(feature_maps)   # (b, 1, 28, 28)
            # elif num == 2:
            #     attention_map = self.layer3_conv1_1(feature_maps)   # (b, 1, 14, 14)
            # elif num == 3:
            if num == 3:
                # attention_map = self.layer4_conv1_1(feature_maps)   # (b, 1, 14, 14)
                attention_map = self.avgpool1(feature_maps)
                attention_map = attention_map.view(attention_map.shape[0], -1)  # (b, 1024)
                attention_map = self.attn_fc2(self.relu(self.attn_fc1(attention_map)))
                # attention_map = self.sigmoid(attention_map)
                attention_map = (attention_map - torch.min(attention_map, 1)[0][:, None]) / (1e-6 + torch.max(attention_map, 1)[0][:, None] - torch.min(attention_map, 1)[0][:, None])
                attention_map = attention_map[:, :, None, None] * feature_maps   # (b, 1024, 14, 14)
                attention_map = torch.mean(attention_map, 1)  # (b, 14, 14)

                # x = attention_map * feature_maps   # (b, 256, 56, 56)
                x = attention_map[:, None, :, :] * feature_maps   # (b, 256, 56, 56)
                x = x.view(x.shape[0], x.shape[1], -1)   # (b, 256, 56*56)
                x = x.permute(0, 2, 1)   # (b, 56*56, 256)

            num += 1

        x = self.swinB.norm(x)  # B L C
        x = self.swinB.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        outputs_ori = self.fc(x)

        # # print('the shape of the all_hidden_features is :', all_hidden_features.shape)   # (b, 12, 768)
        # LSTM_inputs = all_hidden_features
        #
        # for i in range(12):
        #     if i == 0:
        #         output, hidden, cell = self.LSTMcell(input=LSTM_inputs[:, i, :], hidden=torch.ones(x.shape[0], 768).cuda(), cell=torch.ones(x.shape[0], self.num_classes).cuda())
        #     else:
        #         output, hidden, cell = self.LSTMcell(input=LSTM_inputs[:, i, :], hidden=hidden, cell=cell)
        #
        # outputs_LSTM = output   # (b, c)

        # return outputs_ori, outputs_LSTM
        return outputs_ori, outputs_ori, attention_map


class mytry5_20220512(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220512, self).__init__()
        self.swinB = transformer
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(1024, num_classes)
        self.num_classes = num_classes

    def forward(self, x, need_feature=False):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)

        for layer in self.swinB.layers:
            x = layer(x)

        x = self.swinB.norm(x)  # B L C
        x = self.swinB.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x7 = self.fc1(x)
        return x7



class mytry5_20220519_v3(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220519_v3, self).__init__()
        self.swinB = transformer
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # self.myLSTM = LSTMCell_mytry5_20220519_v2(hidden_size=2048, cell_size=2048, output_size=2048, num_classes=num_classes)
        # self.myLSTM = nn.LSTM(input_size=1024, hidden_size=num_classes, num_layers=1, batch_first=True)

        self.Upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)   # (56, 56)->(28, 28)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False)  # (28, 28)->(14, 14)
        self.bn1_2 = nn.BatchNorm2d(1024)
        self.conv1_3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(1024)

        self.conv2_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False)   # (28, 28)->(14, 14)
        self.bn2_1 = nn.BatchNorm2d(1024)
        self.conv2_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(1024)

        self.conv3_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(1024)

        self.conv4_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(1024)

        self.Q1 = nn.Linear(1024, 1024)
        self.K1 = nn.Linear(1024, 1024)
        self.V1 = nn.Linear(1024, 1024)
        self.Q2 = nn.Linear(1024, 1024)
        self.K2 = nn.Linear(1024, 1024)
        self.V2 = nn.Linear(1024, 1024)
        self.Q3 = nn.Linear(1024, 1024)
        self.K3 = nn.Linear(1024, 1024)
        self.V3 = nn.Linear(1024, 1024)
        self.Q12 = nn.Linear(1024, 1024)
        self.K12 = nn.Linear(1024, 1024)
        self.V12 = nn.Linear(1024, 1024)
        self.Q123 = nn.Linear(1024, 1024)
        self.K123 = nn.Linear(1024, 1024)
        self.V123 = nn.Linear(1024, 1024)
        self.Q4 = nn.Linear(1024, 1024)
        self.K4 = nn.Linear(1024, 1024)
        self.V4 = nn.Linear(1024, 1024)

        self.fc0 = nn.Linear(1024, num_classes)
        # self.fc1 = nn.Linear(1024, num_classes)
        # self.fc2 = nn.Linear(1024, num_classes)
        # self.fc3 = nn.Linear(1024, num_classes)
        # self.fc4 = nn.Linear(1024, num_classes)
        self.num_classes = num_classes

        # self.fc4 = nn.Linear(1024, num_classes)
        # self.gate3_fc1 = nn.Linear(1024, num_classes)
        # self.gate3_K = nn.Linear(num_classes, num_classes)
        # self.gate3_V = nn.Linear(num_classes, num_classes)
        # self.gate3_Q = nn.Linear(num_classes, num_classes)
        # self.gate2_fc1 = nn.Linear(1024, num_classes)
        # self.gate2_K = nn.Linear(num_classes, num_classes)
        # self.gate2_V = nn.Linear(num_classes, num_classes)
        # self.gate2_Q = nn.Linear(num_classes, num_classes)
        # self.gate1_fc1 = nn.Linear(1024, num_classes)
        # self.gate1_K = nn.Linear(num_classes, num_classes)
        # self.gate1_V = nn.Linear(num_classes, num_classes)
        # self.gate1_Q = nn.Linear(num_classes, num_classes)
        #
        # self.gate3_bn = nn.BatchNorm1d(num_classes)
        # self.gate2_bn = nn.BatchNorm1d(num_classes)
        # self.gate1_bn = nn.BatchNorm1d(num_classes)

        # self.feature_MLP_fc1 = nn.Linear(1024, 1024)
        # self.feature_MLP_fc2 = nn.Linear(1024, 1024)
        # self.feature_MLP_fc3 = nn.Linear(1024, num_classes)

        self.MLP1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

        # ########## for the myLSTMcell ######################
        self.LSTM_fc = nn.Linear(2048, 1024)
        self.tanh = nn.Tanh()
        # ####################################################
        self.LSTM_output_fc = nn.Linear(1024, num_classes)

    def forward(self, x, need_feature=False):
        x = self.swinB.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.swinB.pos_drop(x)
        # print(self.swinB.num_features)   # (1024)

        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            if layer_num == 0:
                x1 = x
                x1 = x1.permute(0, 2, 1)
                x1 = x1.view(x1.shape[0], x1.shape[1], int(math.sqrt(x1.shape[2])), int(math.sqrt(x1.shape[2])))
            elif layer_num == 1:
                x2 = x
                x2 = x2.permute(0, 2, 1)
                x2 = x2.view(x2.shape[0], x2.shape[1], int(math.sqrt(x2.shape[2])), int(math.sqrt(x2.shape[2])))
            elif layer_num == 2:
                x3 = x
                x3 = x3.permute(0, 2, 1)
                x3 = x3.view(x3.shape[0], x3.shape[1], int(math.sqrt(x3.shape[2])), int(math.sqrt(x3.shape[2])))
            elif layer_num == 3:
                x4 = x
                x4 = x4.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], x4.shape[1], int(math.sqrt(x4.shape[2])), int(math.sqrt(x4.shape[2])))

            layer_num = layer_num + 1

        x = self.swinB.norm(x)  # B L C
        x = self.swinB.avgpool(x.transpose(1, 2))  # B C 1
        # print(x.shape)  # (b, 1024, 1)
        x = torch.flatten(x, 1)

        logits0 = self.fc0(x)
        logits0_ori = logits0

        # features = self.feature_MLP_fc3(self.relu(self.feature_MLP_fc2(self.relu(self.feature_MLP_fc1(x)))))
        # norm_features = nn.functional.normalize(features, p=2, dim=1)

        F1 = x4
        F2 = x3

        F3 = x2
        F4 = x1

        F1 = self.relu(self.bn4_1(self.conv4_1(F1)))
        F2 = self.relu(self.bn3_1(self.conv3_1(F2)))
        F3 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(F3))))))
        F4 = self.relu(self.bn1_3(self.conv1_3(self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(F4)))))))))

        F1 = self.maxpool1(F1).view(x.shape[0], -1)
        F2 = self.maxpool1(F2).view(x.shape[0], -1)
        F3 = self.maxpool1(F3).view(x.shape[0], -1)
        F4 = self.maxpool1(F4).view(x.shape[0], -1)  # (b, 1024)

        F1_ = F1
        Q_F1_, K_F1_, V_F1_ = self.Q1(F1_), self.K1(F1_), self.V1(F1_)
        Q_F2, K_F2, V_F2 = self.Q2(F2), self.K2(F2), self.V2(F2)
        Attned_F1 = torch.mm(torch.softmax(torch.mm(Q_F2, K_F1_.permute(1, 0)) / math.sqrt(1024), 1), V_F1_)
        Attned_F2 = torch.mm(torch.softmax(torch.mm(Q_F1_, K_F2.permute(1, 0)) / math.sqrt(1024), 1), V_F2)
        F12 = Attned_F1 + Attned_F2
        Q_F3, K_F3, V_F3 = self.Q3(F3), self.K3(F3), self.V3(F3)
        Q_F12, K_F12, V_F12 = self.Q12(F12), self.K12(F12), self.V12(F12)
        Attned_F3 = torch.mm(torch.softmax(torch.mm(Q_F12, K_F3.permute(1, 0)) / math.sqrt(1024), 1), V_F3)
        Attned_F12 = torch.mm(torch.softmax(torch.mm(Q_F3, K_F12.permute(1, 0)) / math.sqrt(1024), 1), V_F12)
        F123 = Attned_F3 + Attned_F12
        Q_F4, K_F4, V_F4 = self.Q4(F4), self.K4(F4), self.V4(F4)
        Q_F123, K_F123, V_F123 = self.Q123(F123), self.K123(F123), self.V123(F123)
        Attned_F4 = torch.mm(torch.softmax(torch.mm(Q_F123, K_F4.permute(1, 0)) / math.sqrt(1024), 1), V_F4)
        Attned_F123 = torch.mm(torch.softmax(torch.mm(Q_F4, K_F123.permute(1, 0)) / math.sqrt(1024), 1), V_F123)
        F1234 = Attned_F4 + Attned_F123

        Hidden = self.MLP1(F1)
        Cell = self.MLP2(F1)

        LSTM_output1, Hidden1, Cell1 = self.myLSTMcell(Hidden, Cell, F1234)
        LSTM_output2, Hidden2, Cell2 = self.myLSTMcell(Hidden1, Cell1, F123)
        LSTM_output3, Hidden3, Cell3 = self.myLSTMcell(Hidden2, Cell2, F12)
        LSTM_output4, Hidden4, Cell4 = self.myLSTMcell(Hidden3, Cell3, F1)

        LSTM_logits = self.LSTM_output_fc(LSTM_output4)

        # return logits0_ori, logits1, logits2, logits3, logits4, F1, F12, F123, F1234
        # return logits0_ori, norm_features
        return logits0_ori, LSTM_logits

    def myLSTMcell(self, Hidden, Cell, input):
        Cell_ = Cell * self.sigmoid(self.LSTM_fc(torch.cat((Hidden, input), 1)))
        Cell_ = Cell_ + self.sigmoid(self.LSTM_fc(torch.cat((Hidden, input), 1))) * self.tanh(self.LSTM_fc(torch.cat((Hidden, input), 1)))
        output = self.tanh(Cell_) * self.sigmoid(self.LSTM_fc(torch.cat((Hidden, input), 1)))
        Cell = Cell_
        Hidden = output
        return output, Hidden, Cell


class mytry5_20220531_v9_4(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220531_v9_4, self).__init__()
        self.swinB = transformer
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.Upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)   # (56, 56)->(28, 28)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False)  # (28, 28)->(14, 14)
        self.bn1_2 = nn.BatchNorm2d(1024)
        self.conv1_3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(1024)

        self.conv2_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False)   # (28, 28)->(14, 14)
        self.bn2_1 = nn.BatchNorm2d(1024)
        self.conv2_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(1024)

        self.conv3_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(1024)

        self.conv4_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(1024)

        self.Q1 = nn.Linear(1024, 1024)
        self.K1 = nn.Linear(1024, 1024)
        self.V1 = nn.Linear(1024, 1024)
        self.Q2 = nn.Linear(1024, 1024)
        self.K2 = nn.Linear(1024, 1024)
        self.V2 = nn.Linear(1024, 1024)
        self.Q3 = nn.Linear(1024, 1024)
        self.K3 = nn.Linear(1024, 1024)
        self.V3 = nn.Linear(1024, 1024)
        self.Q12 = nn.Linear(1024, 1024)
        self.K12 = nn.Linear(1024, 1024)
        self.V12 = nn.Linear(1024, 1024)
        self.Q123 = nn.Linear(1024, 1024)
        self.K123 = nn.Linear(1024, 1024)
        self.V123 = nn.Linear(1024, 1024)
        self.Q4 = nn.Linear(1024, 1024)
        self.K4 = nn.Linear(1024, 1024)
        self.V4 = nn.Linear(1024, 1024)

        self.fc0 = nn.Linear(1024, num_classes)
        self.num_classes = num_classes

        self.MLP1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

        # ########## for the myLSTMcell ######################
        self.LSTM_fc = nn.Linear(2048, 1024)
        self.LSTM_fc2 = nn.Linear(2048, 1024)
        self.tanh = nn.Tanh()
        # ####################################################
        self.LSTM_output_fc = nn.Linear(1024, num_classes)

        # ######## for the LSTM in the LSTM ##################
        self.LSTM_in = nn.LSTM(input_size=1024+1024, hidden_size=1024, num_layers=1, batch_first=True, bias=False)
        # ####################################################

    def forward(self, x, need_feature=False):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)

        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            if layer_num == 0:
                x1 = x
                x1 = x1.permute(0, 2, 1)
                x1 = x1.view(x1.shape[0], x1.shape[1], int(math.sqrt(x1.shape[2])), int(math.sqrt(x1.shape[2])))
            elif layer_num == 1:
                x2 = x
                x2 = x2.permute(0, 2, 1)
                x2 = x2.view(x2.shape[0], x2.shape[1], int(math.sqrt(x2.shape[2])), int(math.sqrt(x2.shape[2])))
            elif layer_num == 2:
                x3 = x
                x3 = x3.permute(0, 2, 1)
                x3 = x3.view(x3.shape[0], x3.shape[1], int(math.sqrt(x3.shape[2])), int(math.sqrt(x3.shape[2])))
            elif layer_num == 3:
                x4 = x
                x4 = x4.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], x4.shape[1], int(math.sqrt(x4.shape[2])), int(math.sqrt(x4.shape[2])))

            layer_num = layer_num + 1

        x = self.swinB.norm(x)  # B L C
        x = self.swinB.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        logits0 = self.fc0(x)
        logits0_ori = logits0

        F1 = x4
        F2 = x3

        F3 = x2
        F4 = x1

        F1 = self.relu(self.bn4_1(self.conv4_1(F1)))   # (b, 1024, 14, 14)
        F2 = self.relu(self.bn3_1(self.conv3_1(F2)))
        F3 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(F3))))))
        F4 = self.relu(self.bn1_3(self.conv1_3(self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(F4)))))))))

        F1_feature_maps = F1   # (b, 1024, 14, 14)
        F2_feature_maps = F2
        F3_feature_maps = F3
        F4_feature_maps = F4

        F1 = self.maxpool1(F1).view(x.shape[0], -1)
        F2 = self.maxpool1(F2).view(x.shape[0], -1)
        F3 = self.maxpool1(F3).view(x.shape[0], -1)
        F4 = self.maxpool1(F4).view(x.shape[0], -1)  # (b, 1024)

        F1_ = F1
        Q_F1_, K_F1_, V_F1_ = self.Q1(F1_), self.K1(F1_), self.V1(F1_)
        Q_F2, K_F2, V_F2 = self.Q2(F2), self.K2(F2), self.V2(F2)
        Attned_F1 = torch.mm(torch.softmax(torch.mm(Q_F2, K_F1_.permute(1, 0)) / math.sqrt(1024), 1), V_F1_)
        Attned_F2 = torch.mm(torch.softmax(torch.mm(Q_F1_, K_F2.permute(1, 0)) / math.sqrt(1024), 1), V_F2)
        F12 = Attned_F1 + Attned_F2
        Q_F3, K_F3, V_F3 = self.Q3(F3), self.K3(F3), self.V3(F3)
        Q_F12, K_F12, V_F12 = self.Q12(F12), self.K12(F12), self.V12(F12)
        Attned_F3 = torch.mm(torch.softmax(torch.mm(Q_F12, K_F3.permute(1, 0)) / math.sqrt(1024), 1), V_F3)
        Attned_F12 = torch.mm(torch.softmax(torch.mm(Q_F3, K_F12.permute(1, 0)) / math.sqrt(1024), 1), V_F12)
        F123 = Attned_F3 + Attned_F12
        Q_F4, K_F4, V_F4 = self.Q4(F4), self.K4(F4), self.V4(F4)
        Q_F123, K_F123, V_F123 = self.Q123(F123), self.K123(F123), self.V123(F123)
        Attned_F4 = torch.mm(torch.softmax(torch.mm(Q_F123, K_F4.permute(1, 0)) / math.sqrt(1024), 1), V_F4)
        Attned_F123 = torch.mm(torch.softmax(torch.mm(Q_F4, K_F123.permute(1, 0)) / math.sqrt(1024), 1), V_F123)
        F1234 = Attned_F4 + Attned_F123

        Hidden = self.MLP1(F1)
        Cell = self.MLP2(F1)

        F1_splits = F1_feature_maps.view(F1_feature_maps.shape[0], 1024, 196)   # (b, 1024, 196)
        F2_splits = F2_feature_maps.view(F2_feature_maps.shape[0], 1024, 196)   # (b, 1024, 196)
        F3_splits = F3_feature_maps.view(F3_feature_maps.shape[0], 1024, 196)   # (b, 1024, 196)
        F4_splits = F4_feature_maps.view(F4_feature_maps.shape[0], 1024, 196)   # (b, 1024, 196)

        F1_splits = F1_splits.permute(0, 2, 1)   # (b, 196, 1024)
        F2_splits = F2_splits.permute(0, 2, 1)   # (b, 196, 1024)
        F3_splits = F3_splits.permute(0, 2, 1)   # (b, 196, 1024)
        F4_splits = F4_splits.permute(0, 2, 1)   # (b, 196, 1024)

        LSTM_output1, Hidden1, Cell1 = self.myLSTMcell(Hidden, Cell, F1234, F4_splits)
        LSTM_output2, Hidden2, Cell2 = self.myLSTMcell(Hidden1, Cell1, F123, F3_splits)
        LSTM_output3, Hidden3, Cell3 = self.myLSTMcell(Hidden2, Cell2, F12, F2_splits)
        LSTM_output4, Hidden4, Cell4 = self.myLSTMcell(Hidden3, Cell3, F1, F1_splits)

        LSTM_logits = self.LSTM_output_fc(LSTM_output4)

        return logits0_ori, LSTM_logits

    def myLSTMcell(self, Hidden, Cell, input, splits):
        # Cell_ = Cell * self.sigmoid(self.LSTM_fc(torch.cat((Hidden, input), 1)))
        LSTM_in_input = torch.cat((splits, Hidden.unsqueeze(1).repeat(1, 196, 1)), 2)   # (b, 196, 1024+1024)
        LSTM_in_output, (_, _) = self.LSTM_in(LSTM_in_input)   # output: (b, 1024, 1024)
        LSTM_in_output = LSTM_in_output[:, -1, :]   # (b, 1024)
        forget = self.LSTM_fc2(torch.cat((LSTM_in_output, input), 1))
        forget = self.sigmoid(forget)
        Cell_ = Cell * forget
        Cell_ = Cell_ + self.sigmoid(self.LSTM_fc(torch.cat((Hidden, input), 1))) * self.tanh(self.LSTM_fc(torch.cat((Hidden, input), 1)))
        output = self.tanh(Cell_) * self.sigmoid(self.LSTM_fc(torch.cat((Hidden, input), 1)))
        Cell = Cell_
        Hidden = output
        return output, Hidden, Cell


class mytry5_20220901_try6(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try6, self).__init__()
        self.swinB = transformer
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.fc1 = nn.Linear(1024, num_classes)
        self.C1 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        # self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        # self.BN1 = nn.BatchNorm2d(128)
        # self.BN2 = nn.BatchNorm2d(128)

        # self.down = nn.Conv2d(1024, 128, 3, 1, 1, bias=False)
        # self.up = nn.Conv2d(128, 1024, 3, 1, 1, bias=False)
        # self.channel_num = 128
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block = nn.Sequential(
            nn.Conv2d(1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.Cell_conv_block = nn.Sequential(
            nn.Conv2d(1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )

        self.LSTM_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.LSTM_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.LSTM_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.LSTM_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )

        self.add_conv1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.add_conv2 = nn.Sequential(
            nn.Conv2d(256, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.add_conv3 = nn.Sequential(
            nn.Conv2d(512, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.add_conv4 = nn.Sequential(
            nn.Conv2d(1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.channel_conv1 = nn.Sequential(
            nn.Conv2d(256, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.channel_conv2 = nn.Sequential(
            nn.Conv2d(512, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.channel_conv3 = nn.Sequential(
            nn.Conv2d(1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.channel_conv4 = nn.Sequential(
            nn.Conv2d(1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )

        self.AddNorm = nn.LayerNorm(128)

    def LSTM_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        x0 = x.permute(0, 2, 1)
        x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            if layer_num == 1:
                x1 = x.permute(0, 2, 1)
                x1 = x1.view(x1.shape[0], 256, 56, 56)
            elif layer_num == 2:
                x2 = x.permute(0, 2, 1)
                x2 = x2.view(x2.shape[0], 512, 28, 28)
            elif layer_num == 3:
                x3 = x.permute(0, 2, 1)
                x3 = x3.view(x3.shape[0], 1024, 14, 14)
            elif layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################ Upsampling 20220908 ################################
        x_upsample = self.channel_conv4(x4) + self.add_conv4(x3)   # (b, 1024, 14, 14)
        x_upsample = nn.functional.interpolate(x_upsample, size=(28, 28), mode='bilinear')
        x_upsample = self.channel_conv3(x_upsample) + self.add_conv3(x2)   # (b, 512, 28, 28)
        x_upsample = nn.functional.interpolate(x_upsample, size=(56, 56), mode='bilinear')
        x_upsample = self.channel_conv2(x_upsample) + self.add_conv2(x1)   # (b, 256, 56, 56)
        x_upsample = nn.functional.interpolate(x_upsample, size=(112, 112), mode='bilinear')
        x_upsample = self.channel_conv1(x_upsample) + self.add_conv1(x0)   # (b, 128, 112, 112)
        # #################################################################################

        # ori_FM = x  # (b, 1024, 7, 7)
        #
        # frequency_x = torch.rfft(x, 2, onesided=False)
        # frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
        # frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])
        # # filter_high0, filter_low0 = my_filter_square(7, 7, ratio=3 / 4)
        # # filter_high1, filter_low1 = my_filter_square(7, 7, ratio=1 / 2)
        # # filter_high2, filter_low2 = my_filter_square(7, 7, ratio=1 / 4)
        # # filter_high3, filter_low3 = my_filter_square(7, 7, ratio=0)
        # filter_high0, filter_low0 = my_filter_square(14, 14, ratio=3 / 4)
        # filter_high1, filter_low1 = my_filter_square(14, 14, ratio=1 / 2)
        # filter_high2, filter_low2 = my_filter_square(14, 14, ratio=1 / 4)
        # filter_high3, filter_low3 = my_filter_square(14, 14, ratio=0)
        # new_frequency_x0 = frequency_x * filter_high0
        # new_frequency_x0 = (new_frequency_x0 + frequency_x) / 2
        # new_frequency_x0 = torch.cat((new_frequency_x0.real.unsqueeze(4), new_frequency_x0.imag.unsqueeze(4)), 4)
        # x0 = torch.abs(torch.irfft(new_frequency_x0, 2, onesided=False))
        # new_frequency_x1 = frequency_x * filter_high1
        # new_frequency_x1 = (new_frequency_x1 + frequency_x) / 2
        # new_frequency_x1 = torch.cat((new_frequency_x1.real.unsqueeze(4), new_frequency_x1.imag.unsqueeze(4)), 4)
        # x1 = torch.abs(torch.irfft(new_frequency_x1, 2, onesided=False))
        # new_frequency_x2 = frequency_x * filter_high2
        # new_frequency_x2 = (new_frequency_x2 + frequency_x) / 2
        # new_frequency_x2 = torch.cat((new_frequency_x2.real.unsqueeze(4), new_frequency_x2.imag.unsqueeze(4)), 4)
        # x2 = torch.abs(torch.irfft(new_frequency_x2, 2, onesided=False))
        # new_frequency_x3 = frequency_x * filter_high3
        # new_frequency_x3 = (new_frequency_x3 + frequency_x) / 2
        # new_frequency_x3 = torch.cat((new_frequency_x3.real.unsqueeze(4), new_frequency_x3.imag.unsqueeze(4)), 4)
        # x3 = torch.abs(torch.irfft(new_frequency_x3, 2, onesided=False))
        #
        # # Hidden0 = torch.randn(x.shape[0], 1024, 7, 7).cuda()
        # # Cell0 = torch.randn(x.shape[0], 1024, 7, 7).cuda()
        # Hidden0 = self.Hidden_conv_block(x3)
        # Cell0 = self.Cell_conv_block(x3)
        # # Hidden0 = self.Hidden0_state
        # # Cell0 = self.Cell0_state
        # Input0 = x0
        #
        # Output1, Hidden1, Cell1 = self.LSTM_cell(Input0, Hidden0, Cell0)
        # Input1 = x1
        # Output2, Hidden2, Cell2 = self.LSTM_cell(Input1, Hidden1, Cell1)
        # Input2 = x2
        # Output3, Hidden3, Cell3 = self.LSTM_cell(Input2, Hidden2, Cell2)
        # Input3 = x3
        # Output4, Hidden4, Cell4 = self.LSTM_cell(Input3, Hidden3, Cell3)
        #
        # recon_FM = Output4
        #
        # # x = Output4.view(Output4.shape[0], 1024, 49)
        # # x = x.permute(0, 2, 1)
        # #
        # # x = self.swinB.norm(x)  # B L C
        # # x = self.swinB.avgpool(x.transpose(1, 2))  # B C 1
        # # x = torch.flatten(x, 1)
        # # x7 = self.fc1(x)
        #
        # # ori_FM_output = ori_FM.view(ori_FM.shape[0], 1024, 49)
        # ori_FM_output = ori_FM.view(ori_FM.shape[0], 1024, 14*14)
        # ori_FM_output = ori_FM_output.permute(0, 2, 1)
        # ori_FM_output = self.swinB.norm(ori_FM_output)  # B L C
        # ori_FM_output = self.swinB.avgpool(ori_FM_output.transpose(1, 2))  # B C 1
        # ori_FM_output = torch.flatten(ori_FM_output, 1)
        # ori_FM_output = self.fc1(ori_FM_output)
        #
        # # recon_FM_output = recon_FM.view(recon_FM.shape[0], 1024, 49)
        # recon_FM_output = recon_FM.view(recon_FM.shape[0], 1024, 14*14)
        # recon_FM_output = recon_FM_output.permute(0, 2, 1)
        # recon_FM_output = self.swinB.norm(recon_FM_output)  # B L C
        # recon_FM_output = self.swinB.avgpool(recon_FM_output.transpose(1, 2))  # B C 1
        # recon_FM_output = torch.flatten(recon_FM_output, 1)
        # recon_FM_output = self.fc1(recon_FM_output)

        x_upsample = x_upsample.view(x_upsample.shape[0], 128, 112*112)
        x_upsample = x_upsample.permute(0, 2, 1)
        x_upsample = self.AddNorm(x_upsample)  # B L C
        x_upsample = self.swinB.avgpool(x_upsample.transpose(1, 2))  # B C 1
        x_upsample = torch.flatten(x_upsample, 1)
        concat_output = self.C1(x_upsample)

        # return x7
        # return ori_FM_output, recon_FM_output
        return concat_output
        # return low_output, high_mask_output, high_invmask_output


class mytry5_20220901_try7(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        self.C2 = nn.Linear(128, num_classes)
        self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # self.add_conv1 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.add_conv2 = nn.Sequential(
        #     nn.Conv2d(256, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(256)
        # )
        # self.add_conv3 = nn.Sequential(
        #     nn.Conv2d(512, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(512)
        # )
        # self.add_conv4 = nn.Sequential(
        #     nn.Conv2d(1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(1024)
        # )
        # self.channel_conv1 = nn.Sequential(
        #     nn.Conv2d(256, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.channel_conv2 = nn.Sequential(
        #     nn.Conv2d(512, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(256)
        # )
        # self.channel_conv3 = nn.Sequential(
        #     nn.Conv2d(1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(512)
        # )
        # self.channel_conv4 = nn.Sequential(
        #     nn.Conv2d(1024, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(1024)
        # )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        self.AddNorm2 = nn.LayerNorm(128)
        self.AddNorm3 = nn.LayerNorm(128)

        self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)  # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        ori_output = ori_output.permute(0, 2, 1)
        ori_output = self.AddNorm1(ori_output)  # B L C
        ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        ori_output = torch.flatten(ori_output, 1)
        ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        p_params = self.learned_ps  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        high_new_x_all_T = torch.zeros(x_upsample.shape[0], 10, 128, 56, 56).cuda()  # (b, num_T=10, 128, 112, 112)
        low_new_x_all_T = torch.zeros(x_upsample.shape[0], 10, 128, 56, 56).cuda()  # (b, num_T=10, 128, 112, 112)
        for i in range(10):
            # print(i)
            p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            EP_filter_low = 1. - EP_filter_high  # (b, 128, 112, 112)

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

            high_new_x_all_T[:, i, :, :, :] = high_new_x_now  # from the full-pass to high-pass
            low_new_x_all_T[:, i, :, :, :] = low_new_x_now  # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        high_output = high_output.permute(0, 2, 1)
        high_output = self.AddNorm2(high_output)  # B L C
        high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        high_output = torch.flatten(high_output, 1)
        high_output = self.C3(high_output)

        low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        low_output = low_output.permute(0, 2, 1)
        low_output = self.AddNorm3(low_output)  # B L C
        low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        low_output = torch.flatten(low_output, 1)
        low_output = self.C2(low_output)
        # ########################################

        return ori_output, low_output, high_output


class mytry5_20220901_try7_v1(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        # p_params = self.learned_ps  # (1, 128)
        p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            EP_filter_low = 1. - EP_filter_high  # (b, 128, 112, 112)

            if i == 0:
                EP_filter_high_use = EP_filter_high
                EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # high_output = high_output.permute(0, 2, 1)
        # high_output = self.AddNorm2(high_output)  # B L C
        # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # high_output = torch.flatten(high_output, 1)
        # high_output = self.C3(high_output)
        #
        # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # low_output = low_output.permute(0, 2, 1)
        # low_output = self.AddNorm3(low_output)  # B L C
        # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # low_output = torch.flatten(low_output, 1)
        # low_output = self.C2(low_output)

        frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        frequency_high_x_ori = torch.roll(frequency_high_x_ori,
                                          (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2),
                                          dims=(2, 3))
        frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        frequency_low_x_ori = torch.roll(frequency_low_x_ori,
                                         (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2),
                                         dims=(2, 3))
        frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        new_frequency_x = frequency_high_x + frequency_low_x
        new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        new_output = new_x.view(new_x.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try7_v2(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v2, self).__init__()
        self.swinB = transformer
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)  # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7_v2 20220913 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        p_params = self.learned_ps  # (1, 128)
        # p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            EP_filter_low = 1. - EP_filter_high  # (b, 128, 112, 112)

            if i == 0:
                EP_filter_high_use = EP_filter_high
                EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            if i == 0:
                high_new_x_use = high_new_x_now
                low_new_x_use = low_new_x_now

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        high_use_output = high_new_x_use.view(high_new_x_use.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        high_use_output = high_use_output.permute(0, 2, 1)
        high_use_output = self.AddNorm1(high_use_output)  # B L C
        high_use_output = self.swinB.avgpool(high_use_output.transpose(1, 2))  # B C 1
        high_use_output = torch.flatten(high_use_output, 1)
        high_use_output = self.C1(high_use_output)

        low_use_output = low_new_x_use.view(low_new_x_use.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        low_use_output = low_use_output.permute(0, 2, 1)
        low_use_output = self.AddNorm1(low_use_output)  # B L C
        low_use_output = self.swinB.avgpool(low_use_output.transpose(1, 2))  # B C 1
        low_use_output = torch.flatten(low_use_output, 1)
        low_use_output = self.C1(low_use_output)

        ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        ori_output = ori_output.permute(0, 2, 1)
        ori_output = self.AddNorm1(ori_output)  # B L C
        ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        ori_output = torch.flatten(ori_output, 1)
        ori_output = self.C1(ori_output)

        frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        frequency_high_x_ori = torch.roll(frequency_high_x_ori,
                                          (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2),
                                          dims=(2, 3))
        frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        frequency_low_x_ori = torch.roll(frequency_low_x_ori,
                                         (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2),
                                         dims=(2, 3))
        frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        new_frequency_x = frequency_high_x + frequency_low_x
        new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        new_x_high = torch.abs(torch.irfft(frequency_high_x, 2, onesided=False))
        new_x_low = torch.abs(torch.irfft(frequency_low_x, 2, onesided=False))

        new_output = new_x.view(new_x.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        new_high_output = new_x_high.view(new_x_high.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        new_high_output = new_high_output.permute(0, 2, 1)
        new_high_output = self.AddNorm1(new_high_output)  # B L C
        new_high_output = self.swinB.avgpool(new_high_output.transpose(1, 2))  # B C 1
        new_high_output = torch.flatten(new_high_output, 1)
        new_high_output = self.C1(new_high_output)

        new_low_output = new_x_low.view(new_x_low.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        new_low_output = new_low_output.permute(0, 2, 1)
        new_low_output = self.AddNorm1(new_low_output)  # B L C
        new_low_output = self.swinB.avgpool(new_low_output.transpose(1, 2))  # B C 1
        new_low_output = torch.flatten(new_low_output, 1)
        new_low_output = self.C1(new_low_output)

        # ########################################

        # return ori_output, low_output, high_output
        # return new_output
        # return ori_output, high_use_output, low_use_output, new_output, new_high_output, new_low_output
        return ori_output, new_output


class mytry5_20220901_try7_v1_p1(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1_p1, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        # self.Hidden_conv_block_LSTM1 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM1 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        #
        # self.LSTM1_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    # def LSTM1_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell
    #
    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        # Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        # Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        # Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        # Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)
        #
        # all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56*56)  # (1, 20, 112*112)
        # all_templates = all_templates.permute(1, 0, 2)   # (20, 1, 112*112)
        #
        # # p_params = self.learned_ps  # (1, 128)
        # p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        # p_params = nn.functional.softplus(p_params)  # to be positive
        # p_params = p_params.clamp(0, 20)
        # sigma_EP = 20. * torch.ones(128, 20).cuda()   # (128, 20)
        # x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        # x_EP = x_EP + 0.5
        # x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        # p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)
        #
        # # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # # for i in range(10):
        # for i in range(4):
        #     # print(i)
        #     # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
        #     # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
        #     p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
        #     EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
        #     EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20, 56*56)   # (128, 20, 112*112)
        #     # print(torch.isnan(EP_values).any())
        #
        #     EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2, 0)  # (128, 1, 112*112)
        #     EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
        #     EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)   # (b, 128, 112, 112)
        #     # print(torch.isnan(EP_filter_high).any())
        #
        #     EP_filter_low = 1. - EP_filter_high   # (b, 128, 112, 112)
        #
        #     if i == 0:
        #         EP_filter_high_use = EP_filter_high
        #         EP_filter_low_use = EP_filter_low
        #
        #     frequency_x = torch.rfft(x_upsample, 2, onesided=False)
        #     frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
        #     frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])
        #
        #     high_new_frequency_x = frequency_x * EP_filter_high
        #     high_new_frequency_x = torch.cat((high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
        #     high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
        #     low_new_frequency_x = frequency_x * EP_filter_low
        #     low_new_frequency_x = torch.cat((low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
        #     low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
        #
        #     # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
        #     # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass
        #
        #     # ############ for LSTMs ###############
        #     if i == 0:
        #         LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
        #         LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
        #     else:
        #         LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
        #         LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
        #     # ######################################
        #
        # # ############ for outputs ###############
        # # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # high_output = high_output.permute(0, 2, 1)
        # # high_output = self.AddNorm2(high_output)  # B L C
        # # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # # high_output = torch.flatten(high_output, 1)
        # # high_output = self.C3(high_output)
        # #
        # # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # low_output = low_output.permute(0, 2, 1)
        # # low_output = self.AddNorm3(low_output)  # B L C
        # # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # # low_output = torch.flatten(low_output, 1)
        # # low_output = self.C2(low_output)
        #
        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        new_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try7_v1_p2(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1_p2, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        # Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        # Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        # p_params = self.learned_ps  # (1, 128)
        p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            # EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = torch.exp(- (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            # EP_filter_low = 1. - EP_filter_high   # (b, 128, 112, 112)

            # if i == 0:
            #     EP_filter_high_use = EP_filter_high
            #     # EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_frequency_x = torch.roll(high_new_frequency_x,
                                              (high_new_frequency_x.shape[2] // 2, high_new_frequency_x.shape[3] // 2),
                                              dims=(2, 3))
            # high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
            high_new_x_now = torch.irfft(high_new_frequency_x, 2, onesided=False)  # (b, 128, 112, 112)
            # low_new_frequency_x = frequency_x * EP_filter_low
            # low_new_frequency_x = torch.cat((low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            # low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                # LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                # LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # high_output = high_output.permute(0, 2, 1)
        # high_output = self.AddNorm2(high_output)  # B L C
        # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # high_output = torch.flatten(high_output, 1)
        # high_output = self.C3(high_output)
        #
        # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # low_output = low_output.permute(0, 2, 1)
        # low_output = self.AddNorm3(low_output)  # B L C
        # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # low_output = torch.flatten(low_output, 1)
        # low_output = self.C2(low_output)

        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        new_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try7_v1_p3(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1_p3, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        # self.Hidden_conv_block_LSTM1 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM1 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # self.LSTM1_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    # def LSTM1_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        # Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        # Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        # p_params = self.learned_ps  # (1, 128)
        p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            # EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = torch.exp(- (torch.abs(x_EP - 20) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_low = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                             0)  # (128, 1, 112*112)
            EP_filter_low = torch.squeeze(EP_filter_low).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_low = EP_filter_low.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            # EP_filter_low = 1. - EP_filter_high   # (b, 128, 112, 112)

            if i == 0:
                # EP_filter_high_use = EP_filter_high
                EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            # high_new_frequency_x = frequency_x * EP_filter_high
            # high_new_frequency_x = torch.cat((high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            # high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_frequency_x = torch.roll(low_new_frequency_x,
                                             (low_new_frequency_x.shape[2] // 2, low_new_frequency_x.shape[3] // 2),
                                             dims=(2, 3))
            # low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
            low_new_x_now = torch.irfft(low_new_frequency_x, 2, onesided=False)  # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                # LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                # LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # high_output = high_output.permute(0, 2, 1)
        # high_output = self.AddNorm2(high_output)  # B L C
        # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # high_output = torch.flatten(high_output, 1)
        # high_output = self.C3(high_output)
        #
        # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # low_output = low_output.permute(0, 2, 1)
        # low_output = self.AddNorm3(low_output)  # B L C
        # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # low_output = torch.flatten(low_output, 1)
        # low_output = self.C2(low_output)

        # # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # # new_frequency_x = frequency_high_x + frequency_low_x
        # # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        new_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try7_v1_p4(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1_p4, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.C1 = nn.Linear(128, num_classes)
        self.C1 = nn.Linear(256, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        # self.AddNorm1 = nn.LayerNorm(128)
        self.AddNorm1 = nn.LayerNorm(256)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        # p_params = self.learned_ps  # (1, 128)
        p_params_high = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params_high = nn.functional.softplus(p_params_high)  # to be positive
        p_params_high = p_params_high.clamp(0, 20)
        p_params_low = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params_low = nn.functional.softplus(p_params_low)  # to be positive
        p_params_low = p_params_low.clamp(0, 20)

        # #############################################################################
        # ######################## for testing under a fixed p ########################

        p_params_high = 19.9 * torch.ones_like(p_params_high)
        p_params_low = 19.9 * torch.ones_like(p_params_low)

        # #############################################################################

        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP_high = torch.squeeze(p_params_high).unsqueeze(1).repeat(1, 20)  # (128, 20)
        p_EP_low = torch.squeeze(p_params_low).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP_high) * (4 - 1 - i) / (4 - 1)
            # EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = torch.exp(- (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            # EP_filter_low = 1. - EP_filter_high   # (b, 128, 112, 112)
            p_EP_now = 20 - (20 - p_EP_low) * (4 - 1 - i) / (4 - 1)
            # EP_values = (torch.exp(- (torch.abs(x_EP - 20) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = torch.exp(- (torch.abs(x_EP - 20) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            EP_filter_low = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                             0)  # (128, 1, 112*112)
            EP_filter_low = torch.squeeze(EP_filter_low).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_low = EP_filter_low.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)

            if i == 0:
                EP_filter_high_use = EP_filter_high
                EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_frequency_x = torch.roll(high_new_frequency_x,
                                              (high_new_frequency_x.shape[2] // 2, high_new_frequency_x.shape[3] // 2),
                                              dims=(2, 3))
            # high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
            high_new_x_now = torch.irfft(high_new_frequency_x, 2, onesided=False)  # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_frequency_x = torch.roll(low_new_frequency_x,
                                             (low_new_frequency_x.shape[2] // 2, low_new_frequency_x.shape[3] // 2),
                                             dims=(2, 3))
            # low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
            low_new_x_now = torch.irfft(low_new_frequency_x, 2, onesided=False)  # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # high_output = high_output.permute(0, 2, 1)
        # high_output = self.AddNorm2(high_output)  # B L C
        # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # high_output = torch.flatten(high_output, 1)
        # high_output = self.C3(high_output)
        #
        # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # low_output = low_output.permute(0, 2, 1)
        # low_output = self.AddNorm3(low_output)  # B L C
        # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # low_output = torch.flatten(low_output, 1)
        # low_output = self.C2(low_output)

        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        new_x = torch.cat((LSTM1_output, LSTM2_output), 1)  # (b, 256, 112, 112)

        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        new_output = new_x.view(new_x.shape[0], 256, 56 * 56)  # (b, 128, 112*112)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try7_v1_p5(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1_p5, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C1 = nn.Linear(256, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm1 = nn.LayerNorm(256)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        # p_params = self.learned_ps  # (1, 128)
        p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            EP_filter_low = 1. - EP_filter_high  # (b, 128, 112, 112)

            if i == 0:
                EP_filter_high_use = EP_filter_high
                EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # high_output = high_output.permute(0, 2, 1)
        # high_output = self.AddNorm2(high_output)  # B L C
        # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # high_output = torch.flatten(high_output, 1)
        # high_output = self.C3(high_output)
        #
        # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # low_output = low_output.permute(0, 2, 1)
        # low_output = self.AddNorm3(low_output)  # B L C
        # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # low_output = torch.flatten(low_output, 1)
        # low_output = self.C2(low_output)

        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        # new_x = torch.cat((LSTM1_output, LSTM2_output), 1)   # (b, 256, 112, 112)
        new_x = LSTM1_output + LSTM2_output

        new_output = new_x.view(new_x.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        # new_output = new_x.view(new_x.shape[0], 256, 56 * 56)   # (b, 128, 112*112)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try7_v1_p6(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1_p6, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.C1 = nn.Linear(128, num_classes)
        # self.C1 = nn.Linear(256, num_classes)
        self.C1 = nn.Linear(384, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
        )

        # self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm1 = nn.LayerNorm(256)
        self.AddNorm1 = nn.LayerNorm(384)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        # p_params = self.learned_ps  # (1, 128)
        p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            EP_filter_low = 1. - EP_filter_high  # (b, 128, 112, 112)

            if i == 0:
                EP_filter_high_use = EP_filter_high
                EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # high_output = high_output.permute(0, 2, 1)
        # high_output = self.AddNorm2(high_output)  # B L C
        # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # high_output = torch.flatten(high_output, 1)
        # high_output = self.C3(high_output)
        #
        # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # low_output = low_output.permute(0, 2, 1)
        # low_output = self.AddNorm3(low_output)  # B L C
        # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # low_output = torch.flatten(low_output, 1)
        # low_output = self.C2(low_output)

        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        new_x = torch.cat((x_upsample, LSTM1_output, LSTM2_output), 1)  # (b, 256, 112, 112)

        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # new_output = new_x.view(new_x.shape[0], 256, 56 * 56)   # (b, 128, 112*112)
        new_output = new_x.view(new_x.shape[0], 384, 56 * 56)  # (b, 128, 112*112)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try7_v1_p7(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1_p7, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C1 = nn.Linear(256, num_classes)
        # self.C1 = nn.Linear(384, num_classes)
        self.C2 = nn.Linear(128, num_classes)
        self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm1 = nn.LayerNorm(256)
        # self.AddNorm1 = nn.LayerNorm(384)
        self.AddNorm2 = nn.LayerNorm(128)
        self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.weights = nn.Parameter(torch.randn(3).cuda(), requires_grad=True)  # [3]
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        # p_params = self.learned_ps  # (1, 128)
        p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            EP_filter_low = 1. - EP_filter_high  # (b, 128, 112, 112)

            if i == 0:
                EP_filter_high_use = EP_filter_high
                EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # high_output = high_output.permute(0, 2, 1)
        # high_output = self.AddNorm2(high_output)  # B L C
        # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # high_output = torch.flatten(high_output, 1)
        # high_output = self.C3(high_output)
        #
        # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # low_output = low_output.permute(0, 2, 1)
        # low_output = self.AddNorm3(low_output)  # B L C
        # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # low_output = torch.flatten(low_output, 1)
        # low_output = self.C2(low_output)

        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        # new_x = torch.cat((x_upsample, LSTM1_output, LSTM2_output), 1)   # (b, 256, 112, 112)

        # # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # new_output = new_x.view(new_x.shape[0], 256, 56 * 56)   # (b, 128, 112*112)
        # new_output = new_x.view(new_x.shape[0], 384, 56 * 56)   # (b, 128, 112*112)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)

        ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        ori_output = ori_output.permute(0, 2, 1)
        ori_output = self.AddNorm1(ori_output)  # B L C
        ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        ori_output = torch.flatten(ori_output, 1)
        ori_output = self.C1(ori_output)

        high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        high_output = high_output.permute(0, 2, 1)
        high_output = self.AddNorm2(high_output)  # B L C
        high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        high_output = torch.flatten(high_output, 1)
        high_output = self.C2(high_output)

        low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        low_output = low_output.permute(0, 2, 1)
        low_output = self.AddNorm3(low_output)  # B L C
        low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        low_output = torch.flatten(low_output, 1)
        low_output = self.C3(low_output)

        all_output = torch.cat((ori_output.unsqueeze(2), high_output.unsqueeze(2), low_output.unsqueeze(2)),
                               2)  # (b, c, 3)
        class_weights = self.weights.unsqueeze(0).unsqueeze(2).repeat(all_output.shape[0], 1, 1)  # (b, 3, 1)

        new_output = torch.matmul(all_output, class_weights)  # (b, c, 1)
        new_output = torch.squeeze(new_output)  # (b, c)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try7_v1_p8(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try7_v1_p8, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C1 = nn.Linear(256, num_classes)
        # self.C1 = nn.Linear(384, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm1 = nn.LayerNorm(256)
        # self.AddNorm1 = nn.LayerNorm(384)
        self.AddNorm2 = nn.LayerNorm(128)
        self.AddNorm3 = nn.LayerNorm(128)

        self.Q_weight = nn.Linear(128, 128)
        self.K_weight = nn.Linear(128, 128)
        self.V_weight = nn.Linear(128, 128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)

        # p_params = self.learned_ps  # (1, 128)
        p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # for i in range(10):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 112*112)
            # print(torch.isnan(EP_values).any())

            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 112*112)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 112, 112)
            # print(torch.isnan(EP_filter_high).any())

            EP_filter_low = 1. - EP_filter_high  # (b, 128, 112, 112)

            if i == 0:
                EP_filter_high_use = EP_filter_high
                EP_filter_low_use = EP_filter_low

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

            # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
            # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        # ############ for outputs ###############
        # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # high_output = high_output.permute(0, 2, 1)
        # high_output = self.AddNorm2(high_output)  # B L C
        # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # high_output = torch.flatten(high_output, 1)
        # high_output = self.C3(high_output)
        #
        # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # low_output = low_output.permute(0, 2, 1)
        # low_output = self.AddNorm3(low_output)  # B L C
        # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # low_output = torch.flatten(low_output, 1)
        # low_output = self.C2(low_output)

        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)

        # new_x = torch.cat((x_upsample, LSTM1_output, LSTM2_output), 1)   # (b, 256, 112, 112)

        ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        ori_output = ori_output.permute(0, 2, 1)
        ori_output = self.AddNorm1(ori_output)  # B L C
        ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        ori_output = torch.flatten(ori_output, 1)

        high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        high_output = high_output.permute(0, 2, 1)
        high_output = self.AddNorm2(high_output)  # B L C
        high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        high_output = torch.flatten(high_output, 1)

        low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)  # (b, 128, 112*112)
        low_output = low_output.permute(0, 2, 1)
        low_output = self.AddNorm3(low_output)  # B L C
        low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        low_output = torch.flatten(low_output, 1)

        Q = self.Q_weight(high_output)
        K = self.K_weight(low_output)
        V = self.V_weight(ori_output)

        Att_weights = torch.softmax((torch.matmul(Q, K.t())) / (math.sqrt(128)), 1)
        new_output = torch.matmul(Att_weights, V)

        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # new_output = new_x.view(new_x.shape[0], 256, 56 * 56)   # (b, 128, 112*112)
        # # new_output = new_x.view(new_x.shape[0], 384, 56 * 56)   # (b, 128, 112*112)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try8_v1(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try8_v1, self).__init__()
        self.swinB = transformer
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.C1 = nn.Linear(128, num_classes)
        self.C1 = nn.Linear(128 * 2, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()
        #
        # self.Hidden_conv_block_LSTM1 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM1 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        #
        # self.LSTM1_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.p_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.high_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.low_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # self.AddNorm1 = nn.LayerNorm(128)
        self.AddNorm1 = nn.LayerNorm(128 * 2)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try8_v1 20220918 ######################################
        p = self.p_net(x_upsample)  # (b, 128, 1, 1)
        p = p.view(p.shape[0], 128)  # (b, 128)
        p = nn.functional.softplus(p)  # to be positive
        p = p.clamp(0, 20)
        sigma_EP = 20. * torch.ones(p.shape[0], 128, 20).cuda()  # (b, 128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).unsqueeze(0).repeat(p.shape[0], 128, 1)  # (b, 128, 20)
        p_EP = torch.squeeze(p).unsqueeze(2).repeat(1, 1, 20)  # (b, 128, 20)
        EP_values = (torch.exp(- (x_EP ** p_EP) / (p_EP * sigma_EP ** p_EP + 1e-6))) / (
                    2 * sigma_EP * p_EP ** (1 / (1e-6 + p_EP)) * torch.exp(
                torch.lgamma(1 + 1 / (1e-6 + p_EP))))  # (b, 128, 20)
        EP_values = EP_values.unsqueeze(3).repeat(1, 1, 1, 56 * 56)  # (b, 128, 20, 56*56)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 56*56)
        all_templates = all_templates.permute(1, 0, 2).unsqueeze(0).repeat(p.shape[0], 1, 1, 1)  # (b, 20, 1, 56*56)

        EP_filter_high = torch.squeeze(
            torch.matmul(EP_values.permute(0, 3, 1, 2), all_templates.permute(0, 3, 1, 2))).permute(0, 2,
                                                                                                    1)  # (b, 128, 56*56)
        EP_filter_high = EP_filter_high.view(EP_filter_high.shape[0], 128, 56, 56)  # (b, 128, 56, 56)
        EP_filter_low = 1. - EP_filter_high  # (b, 128, 56, 56)

        frequency_x = torch.rfft(x_upsample, 2, onesided=False)
        frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
        frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

        high_new_frequency_x = frequency_x * EP_filter_high
        high_new_frequency_x = torch.cat(
            (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
        high_new_x = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 56, 56)

        low_new_frequency_x = frequency_x * EP_filter_low
        low_new_frequency_x = torch.cat((low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)),
                                        4)
        low_new_x = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 56, 56)

        high_new_x_attn = self.high_net(high_new_x)  # (b, 128, 56, 56)
        high_new_x_attn = high_new_x_attn.view(high_new_x_attn.shape[0], 128, 56 * 56)
        high_new_x_attn = torch.softmax(high_new_x_attn, 2)
        high_new_x_attn = high_new_x_attn.view(high_new_x_attn.shape[0], 128, 56, 56)  # (b, 128, 56, 56)
        high_new_x = high_new_x * high_new_x_attn
        low_new_x_attn = self.low_net(low_new_x)  # (b, 128, 56, 56)
        high_new_x = high_new_x + low_new_x_attn

        new_x = torch.cat((high_new_x, low_new_x), 1)  # (b, 128*2, 56, 56)

        new_output = new_x.view(new_x.shape[0], 128 * 2, 56 * 56)  # (b, 128*2, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # #########################################################################################

        # # ################################# try7 20220911 #########################################
        # # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # ori_output = ori_output.permute(0, 2, 1)
        # # ori_output = self.AddNorm1(ori_output)  # B L C
        # # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # # ori_output = torch.flatten(ori_output, 1)
        # # ori_output = self.C1(ori_output)
        #
        # Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        # Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        # Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        # Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)
        #
        # all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56*56)  # (1, 20, 112*112)
        # all_templates = all_templates.permute(1, 0, 2)   # (20, 1, 112*112)
        #
        # # p_params = self.learned_ps  # (1, 128)
        # p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        # p_params = nn.functional.softplus(p_params)  # to be positive
        # p_params = p_params.clamp(0, 20)
        # sigma_EP = 20. * torch.ones(128, 20).cuda()   # (128, 20)
        # x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        # x_EP = x_EP + 0.5
        # x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        # p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)
        #
        # # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # # for i in range(10):
        # for i in range(4):
        #     # print(i)
        #     # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
        #     # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
        #     p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
        #     EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
        #     EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20, 56*56)   # (128, 20, 112*112)
        #     # print(torch.isnan(EP_values).any())
        #
        #     EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2, 0)  # (128, 1, 112*112)
        #     EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
        #     EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)   # (b, 128, 112, 112)
        #     # print(torch.isnan(EP_filter_high).any())
        #
        #     EP_filter_low = 1. - EP_filter_high   # (b, 128, 112, 112)
        #
        #     if i == 0:
        #         EP_filter_high_use = EP_filter_high
        #         EP_filter_low_use = EP_filter_low
        #
        #     frequency_x = torch.rfft(x_upsample, 2, onesided=False)
        #     frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
        #     frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])
        #
        #     high_new_frequency_x = frequency_x * EP_filter_high
        #     high_new_frequency_x = torch.cat((high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
        #     high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
        #     low_new_frequency_x = frequency_x * EP_filter_low
        #     low_new_frequency_x = torch.cat((low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
        #     low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
        #
        #     # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
        #     # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass
        #
        #     # ############ for LSTMs ###############
        #     if i == 0:
        #         LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
        #         LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
        #     else:
        #         LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
        #         LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
        #     # ######################################
        #
        # # ############ for outputs ###############
        # # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # high_output = high_output.permute(0, 2, 1)
        # # high_output = self.AddNorm2(high_output)  # B L C
        # # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # # high_output = torch.flatten(high_output, 1)
        # # high_output = self.C3(high_output)
        # #
        # # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # low_output = low_output.permute(0, 2, 1)
        # # low_output = self.AddNorm3(low_output)  # B L C
        # # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # # low_output = torch.flatten(low_output, 1)
        # # low_output = self.C2(low_output)
        #
        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
        #
        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)
        #
        # # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try8_v2(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try8_v2, self).__init__()
        self.swinB = transformer
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.C1 = nn.Linear(128, num_classes)
        self.C1 = nn.Linear(128 * 2, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128 * 2, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128 * 2)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128 * 2, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128 * 2)
        )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(4 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128 * 2)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(4 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128 * 2)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(4 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128 * 2)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(4 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128 * 2)
        )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.p_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.high_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.low_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # self.AddNorm1 = nn.LayerNorm(128)
        self.AddNorm1 = nn.LayerNorm(128 * 2)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try8_v2 20220918 ######################################
        p = self.p_net(x_upsample)  # (b, 128, 1, 1)
        p = p.view(p.shape[0], 128)  # (b, 128)
        p = nn.functional.softplus(p)  # to be positive
        p = p.clamp(0, 20)
        sigma_EP = 20. * torch.ones(p.shape[0], 128, 20).cuda()  # (b, 128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).unsqueeze(0).repeat(p.shape[0], 128, 1)  # (b, 128, 20)
        p_EP = torch.squeeze(p).unsqueeze(2).repeat(1, 1, 20)  # (b, 128, 20)
        # EP_values = (torch.exp(- (x_EP ** p_EP) / (p_EP * sigma_EP ** p_EP + 1e-6))) / (2 * sigma_EP * p_EP ** (1 / (1e-6 + p_EP)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP))))  # (b, 128, 20)
        # EP_values = EP_values.unsqueeze(3).repeat(1, 1, 1, 56*56)   # (b, 128, 20, 56*56)
        #
        # all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56*56)  # (1, 20, 56*56)
        # all_templates = all_templates.permute(1, 0, 2).unsqueeze(0).repeat(p.shape[0], 1, 1, 1)   # (b, 20, 1, 56*56)
        #
        # EP_filter_high = torch.squeeze(torch.matmul(EP_values.permute(0, 3, 1, 2), all_templates.permute(0, 3, 1, 2))).permute(0, 2, 1)  # (b, 128, 56*56)
        # EP_filter_high = EP_filter_high.view(EP_filter_high.shape[0], 128, 56, 56)  # (b, 128, 56, 56)
        # EP_filter_low = 1. - EP_filter_high   # (b, 128, 56, 56)
        #
        # frequency_x = torch.rfft(x_upsample, 2, onesided=False)
        # frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
        # frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])
        #
        # high_new_frequency_x = frequency_x * EP_filter_high
        # high_new_frequency_x = torch.cat((high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
        # high_new_x = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))   # (b, 128, 56, 56)
        #
        # low_new_frequency_x = frequency_x * EP_filter_low
        # low_new_frequency_x = torch.cat((low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
        # low_new_x = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))   # (b, 128, 56, 56)
        #
        # high_new_x_attn = self.high_net(high_new_x)  # (b, 128, 56, 56)
        # high_new_x_attn = high_new_x_attn.view(high_new_x_attn.shape[0], 128, 56*56)
        # high_new_x_attn = torch.softmax(high_new_x_attn, 2)
        # high_new_x_attn = high_new_x_attn.view(high_new_x_attn.shape[0], 128, 56, 56)  # (b, 128, 56, 56)
        # high_new_x = high_new_x * high_new_x_attn
        # low_new_x_attn = self.low_net(low_new_x)  # (b, 128, 56, 56)
        # high_new_x = high_new_x + low_new_x_attn

        for i in range(4):
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # (b, 128, 20)
            EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (b, 128, 20)
            EP_values = EP_values.unsqueeze(3).repeat(1, 1, 1, 56 * 56)  # (b, 128, 20, 56*56)

            all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 56*56)
            all_templates = all_templates.permute(1, 0, 2).unsqueeze(0).repeat(p.shape[0], 1, 1, 1)  # (b, 20, 1, 56*56)

            EP_filter_high = torch.squeeze(
                torch.matmul(EP_values.permute(0, 3, 1, 2), all_templates.permute(0, 3, 1, 2))).permute(0, 2,
                                                                                                        1)  # (b, 128, 56*56)
            EP_filter_high = EP_filter_high.view(EP_filter_high.shape[0], 128, 56, 56)  # (b, 128, 56, 56)
            EP_filter_low = 1. - EP_filter_high  # (b, 128, 56, 56)

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_x = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 56, 56)

            low_new_frequency_x = frequency_x * EP_filter_low
            low_new_frequency_x = torch.cat(
                (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
            low_new_x = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 56, 56)

            high_new_x_attn = self.high_net(high_new_x)  # (b, 128, 56, 56)
            high_new_x_attn = high_new_x_attn.view(high_new_x_attn.shape[0], 128, 56 * 56)
            high_new_x_attn = torch.softmax(high_new_x_attn, 2)
            high_new_x_attn = high_new_x_attn.view(high_new_x_attn.shape[0], 128, 56, 56)  # (b, 128, 56, 56)
            high_new_x = high_new_x * high_new_x_attn
            low_new_x_attn = self.low_net(low_new_x)  # (b, 128, 56, 56)
            high_new_x = high_new_x + low_new_x_attn

            new_x = torch.cat((high_new_x, low_new_x), 1)  # (b, 128*2=256, 56, 56)

            Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(new_x)  # (b, 256, 56, 56)
            Cell0_LSTM1 = self.Cell_conv_block_LSTM1(new_x)  # (b, 256, 56, 56)

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(new_x, Hidden0_LSTM1, Cell0_LSTM1)
                # LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(new_x, Hidden_LSTM1,
                                                                         Cell_LSTM1)  # (b, 128*2, 56, 56)
                # LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        new_output = LSTM1_output.view(LSTM1_output.shape[0], 128 * 2, 56 * 56)  # (b, 128*2, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # #########################################################################################

        # # ################################# try7 20220911 #########################################
        # # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # ori_output = ori_output.permute(0, 2, 1)
        # # ori_output = self.AddNorm1(ori_output)  # B L C
        # # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # # ori_output = torch.flatten(ori_output, 1)
        # # ori_output = self.C1(ori_output)
        #
        # Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        # Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        # Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        # Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)
        #
        # all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56*56)  # (1, 20, 112*112)
        # all_templates = all_templates.permute(1, 0, 2)   # (20, 1, 112*112)
        #
        # # p_params = self.learned_ps  # (1, 128)
        # p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        # p_params = nn.functional.softplus(p_params)  # to be positive
        # p_params = p_params.clamp(0, 20)
        # sigma_EP = 20. * torch.ones(128, 20).cuda()   # (128, 20)
        # x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        # x_EP = x_EP + 0.5
        # x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        # p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)
        #
        # # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # # for i in range(10):
        # for i in range(4):
        #     # print(i)
        #     # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
        #     # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
        #     p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
        #     EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
        #     EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20, 56*56)   # (128, 20, 112*112)
        #     # print(torch.isnan(EP_values).any())
        #
        #     EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2, 0)  # (128, 1, 112*112)
        #     EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
        #     EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)   # (b, 128, 112, 112)
        #     # print(torch.isnan(EP_filter_high).any())
        #
        #     EP_filter_low = 1. - EP_filter_high   # (b, 128, 112, 112)
        #
        #     if i == 0:
        #         EP_filter_high_use = EP_filter_high
        #         EP_filter_low_use = EP_filter_low
        #
        #     frequency_x = torch.rfft(x_upsample, 2, onesided=False)
        #     frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
        #     frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])
        #
        #     high_new_frequency_x = frequency_x * EP_filter_high
        #     high_new_frequency_x = torch.cat((high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
        #     high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
        #     low_new_frequency_x = frequency_x * EP_filter_low
        #     low_new_frequency_x = torch.cat((low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
        #     low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
        #
        #     # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
        #     # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass
        #
        #     # ############ for LSTMs ###############
        #     if i == 0:
        #         LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
        #         LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
        #     else:
        #         LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
        #         LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
        #     # ######################################
        #
        # # ############ for outputs ###############
        # # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # high_output = high_output.permute(0, 2, 1)
        # # high_output = self.AddNorm2(high_output)  # B L C
        # # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # # high_output = torch.flatten(high_output, 1)
        # # high_output = self.C3(high_output)
        # #
        # # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # low_output = low_output.permute(0, 2, 1)
        # # low_output = self.AddNorm3(low_output)  # B L C
        # # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # # low_output = torch.flatten(low_output, 1)
        # # low_output = self.C2(low_output)
        #
        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
        #
        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)
        #
        # # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try8_v3(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try8_v3, self).__init__()
        self.swinB = transformer
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.C1 = nn.Linear(128, num_classes)
        self.C1 = nn.Linear(128 * 2, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.p_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.high_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.low_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # self.AddNorm1 = nn.LayerNorm(128)
        self.AddNorm1 = nn.LayerNorm(128 * 2)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        templates = torch.zeros(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ZeroPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller))
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.ones(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.zeros(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 112, 112)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try8_v3 20220918 ######################################
        p = self.p_net(x_upsample)  # (b, 128, 1, 1)
        p = p.view(p.shape[0], 128)  # (b, 128)
        p = nn.functional.softplus(p)  # to be positive
        p = p.clamp(0, 20)
        sigma_EP = 20. * torch.ones(p.shape[0], 128, 20).cuda()  # (b, 128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.5
        x_EP = x_EP.unsqueeze(0).unsqueeze(0).repeat(p.shape[0], 128, 1)  # (b, 128, 20)
        p_EP = torch.squeeze(p).unsqueeze(2).repeat(1, 1, 20)  # (b, 128, 20)
        EP_values = (torch.exp(- (x_EP ** p_EP) / (p_EP * sigma_EP ** p_EP + 1e-6))) / (
                    2 * sigma_EP * p_EP ** (1 / (1e-6 + p_EP)) * torch.exp(
                torch.lgamma(1 + 1 / (1e-6 + p_EP))))  # (b, 128, 20)
        EP_values = EP_values.unsqueeze(3).repeat(1, 1, 1, 56 * 56)  # (b, 128, 20, 56*56)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 56*56)
        all_templates = all_templates.permute(1, 0, 2).unsqueeze(0).repeat(p.shape[0], 1, 1, 1)  # (b, 20, 1, 56*56)

        EP_filter_high = torch.squeeze(
            torch.matmul(EP_values.permute(0, 3, 1, 2), all_templates.permute(0, 3, 1, 2))).permute(0, 2,
                                                                                                    1)  # (b, 128, 56*56)
        EP_filter_high = EP_filter_high.view(EP_filter_high.shape[0], 128, 56, 56)  # (b, 128, 56, 56)
        EP_filter_low = 1. - EP_filter_high  # (b, 128, 56, 56)

        frequency_x = torch.rfft(x_upsample, 2, onesided=False)
        frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
        frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

        high_new_frequency_x = frequency_x * EP_filter_high
        high_new_frequency_x = torch.cat(
            (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
        high_new_x = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))  # (b, 128, 56, 56)

        low_new_frequency_x = frequency_x * EP_filter_low
        low_new_frequency_x = torch.cat((low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)),
                                        4)
        low_new_x = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))  # (b, 128, 56, 56)

        high_new_x_attn = self.high_net(high_new_x)  # (b, 128, 56, 56)
        high_new_x_attn = high_new_x_attn.view(high_new_x_attn.shape[0], 128, 56 * 56)
        high_new_x_attn = torch.softmax(high_new_x_attn, 2)
        high_new_x_attn = high_new_x_attn.view(high_new_x_attn.shape[0], 128, 56, 56)  # (b, 128, 56, 56)
        high_new_x = high_new_x * high_new_x_attn
        # low_new_x_attn = self.low_net(low_new_x)  # (b, 128, 56, 56)
        # high_new_x = high_new_x + low_new_x_attn

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(low_new_x)  # (b, 128, 56, 56)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(low_new_x)  # (b, 128, 56, 56)

        for i in range(7):
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            if i < 4:
                p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # (b, 128, 20)
                EP_values = (torch.exp(
                    - (torch.abs(x_EP - 20) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                                        2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                                    torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (b, 128, 20)
            else:
                p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - (i - 3)) / (4 - 1)  # (b, 128, 20)
                EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (
                            2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(
                        torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (b, 128, 20)

            # p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)   # (b, 128, 20)
            # EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (b, 128, 20)
            EP_values = EP_values.unsqueeze(3).repeat(1, 1, 1, 56 * 56)  # (b, 128, 20, 56*56)

            all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 56*56)
            all_templates = all_templates.permute(1, 0, 2).unsqueeze(0).repeat(p.shape[0], 1, 1, 1)  # (b, 20, 1, 56*56)

            EP_filter = torch.squeeze(
                torch.matmul(EP_values.permute(0, 3, 1, 2), all_templates.permute(0, 3, 1, 2))).permute(0, 2,
                                                                                                        1)  # (b, 128, 56*56)
            EP_filter = EP_filter.view(EP_filter.shape[0], 128, 56, 56)  # (b, 128, 56, 56)

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])
            new_frequency_x = frequency_x * EP_filter
            new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 56, 56)

            new_x_attn = self.low_net(new_x)

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(new_x_attn, Hidden0_LSTM1, Cell0_LSTM1)
                # LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(new_x_attn, Hidden_LSTM1,
                                                                         Cell_LSTM1)  # (b, 128*2, 56, 56)
                # LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
            # ######################################

        high_new_x = high_new_x + LSTM1_output
        new_x = torch.cat((high_new_x, low_new_x), 1)  # (b, 128*2, 56*56)

        new_output = new_x.view(new_x.shape[0], 128 * 2, 56 * 56)  # (b, 128*2, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # #########################################################################################

        # # ################################# try7 20220911 #########################################
        # # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # ori_output = ori_output.permute(0, 2, 1)
        # # ori_output = self.AddNorm1(ori_output)  # B L C
        # # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # # ori_output = torch.flatten(ori_output, 1)
        # # ori_output = self.C1(ori_output)
        #
        # Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        # Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        # Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        # Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)
        #
        # all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56*56)  # (1, 20, 112*112)
        # all_templates = all_templates.permute(1, 0, 2)   # (20, 1, 112*112)
        #
        # # p_params = self.learned_ps  # (1, 128)
        # p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        # p_params = nn.functional.softplus(p_params)  # to be positive
        # p_params = p_params.clamp(0, 20)
        # sigma_EP = 20. * torch.ones(128, 20).cuda()   # (128, 20)
        # x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        # x_EP = x_EP + 0.5
        # x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        # p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)
        #
        # # high_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # # low_new_x_all_T = torch.zeros(x_upsample.shape[0], 4, 128, 56, 56).cuda()   # (b, num_T=4, 128, 112, 112)
        # # for i in range(10):
        # for i in range(4):
        #     # print(i)
        #     # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
        #     # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)
        #     p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
        #     EP_values = (torch.exp(- (x_EP ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))) / (2 * sigma_EP * p_EP_now ** (1 / (1e-6 + p_EP_now)) * torch.exp(torch.lgamma(1 + 1 / (1e-6 + p_EP_now))))  # (128, 20)
        #     EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20, 56*56)   # (128, 20, 112*112)
        #     # print(torch.isnan(EP_values).any())
        #
        #     EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2, 0)  # (128, 1, 112*112)
        #     EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 112, 112)
        #     EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)   # (b, 128, 112, 112)
        #     # print(torch.isnan(EP_filter_high).any())
        #
        #     EP_filter_low = 1. - EP_filter_high   # (b, 128, 112, 112)
        #
        #     if i == 0:
        #         EP_filter_high_use = EP_filter_high
        #         EP_filter_low_use = EP_filter_low
        #
        #     frequency_x = torch.rfft(x_upsample, 2, onesided=False)
        #     frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
        #     frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])
        #
        #     high_new_frequency_x = frequency_x * EP_filter_high
        #     high_new_frequency_x = torch.cat((high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
        #     high_new_x_now = torch.abs(torch.irfft(high_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
        #     low_new_frequency_x = frequency_x * EP_filter_low
        #     low_new_frequency_x = torch.cat((low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
        #     low_new_x_now = torch.abs(torch.irfft(low_new_frequency_x, 2, onesided=False))   # (b, 128, 112, 112)
        #
        #     # high_new_x_all_T[:, i, :, :, :] = high_new_x_now   # from the full-pass to high-pass
        #     # low_new_x_all_T[:, i, :, :, :] = low_new_x_now   # from the full-pass to low-pass
        #
        #     # ############ for LSTMs ###############
        #     if i == 0:
        #         LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
        #         LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden0_LSTM2, Cell0_LSTM2)
        #     else:
        #         LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
        #         LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(low_new_x_now, Hidden_LSTM2, Cell_LSTM2)
        #     # ######################################
        #
        # # ############ for outputs ###############
        # # high_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # high_output = high_output.permute(0, 2, 1)
        # # high_output = self.AddNorm2(high_output)  # B L C
        # # high_output = self.swinB.avgpool(high_output.transpose(1, 2))  # B C 1
        # # high_output = torch.flatten(high_output, 1)
        # # high_output = self.C3(high_output)
        # #
        # # low_output = LSTM2_output.view(LSTM2_output.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # # low_output = low_output.permute(0, 2, 1)
        # # low_output = self.AddNorm3(low_output)  # B L C
        # # low_output = self.swinB.avgpool(low_output.transpose(1, 2))  # B C 1
        # # low_output = torch.flatten(low_output, 1)
        # # low_output = self.C2(low_output)
        #
        # frequency_high_x_ori = torch.rfft(LSTM1_output, 2, onesided=False)
        # frequency_high_x_ori = torch.roll(frequency_high_x_ori, (frequency_high_x_ori.shape[2] // 2, frequency_high_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_high_x_ori = torch.complex(frequency_high_x_ori[:, :, :, :, 0], frequency_high_x_ori[:, :, :, :, 1])
        # frequency_low_x_ori = torch.rfft(LSTM2_output, 2, onesided=False)
        # frequency_low_x_ori = torch.roll(frequency_low_x_ori, (frequency_low_x_ori.shape[2] // 2, frequency_low_x_ori.shape[3] // 2), dims=(2, 3))
        # frequency_low_x_ori = torch.complex(frequency_low_x_ori[:, :, :, :, 0], frequency_low_x_ori[:, :, :, :, 1])
        # frequency_high_x = frequency_high_x_ori * EP_filter_high_use
        # frequency_high_x = torch.cat((frequency_high_x.real.unsqueeze(4), frequency_high_x.imag.unsqueeze(4)), 4)
        # frequency_low_x = frequency_low_x_ori * EP_filter_low_use
        # frequency_low_x = torch.cat((frequency_low_x.real.unsqueeze(4), frequency_low_x.imag.unsqueeze(4)), 4)
        # new_frequency_x = frequency_high_x + frequency_low_x
        # new_x = torch.abs(torch.irfft(new_frequency_x, 2, onesided=False))  # (b, 128, 112, 112)
        #
        # new_output = new_x.view(new_x.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)
        #
        # # ########################################

        # return ori_output, low_output, high_output
        return new_output


class mytry5_20220901_try11_v1(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try11_v1, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        self.p_perturbation0 = nn.Parameter(torch.randn(1, 128).cuda())  # (1, 128)
        self.p_perturbation0.requires_grad = True
        self.p_perturbation1 = nn.Parameter(torch.randn(1, 128).cuda())  # (1, 128)
        self.p_perturbation1.requires_grad = True
        self.p_perturbation2 = nn.Parameter(torch.randn(1, 128).cuda())  # (1, 128)
        self.p_perturbation2.requires_grad = True
        self.p_perturbation3 = nn.Parameter(torch.randn(1, 128).cuda())  # (1, 128)
        self.p_perturbation3.requires_grad = True

        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        # print(self.p_perturbation0)
        # print(self.p_perturbation0.requires_grad)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        # Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        # Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)
        # print(torch.max(all_templates))

        # p_params = self.learned_ps  # (1, 128)
        # p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = 0.2 * torch.ones(1, 128).cuda()  # (1, 128)
        p_params = p_params.float()
        # p_params = nn.functional.softplus(p_params)  # to be positive
        # p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20] # [0,1,2,...,19]
        # x_EP = x_EP + 0.5   # [0.5, 1.5, 2.5, ..., 19.5]
        x_EP = x_EP.float()
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        for i in range(4):
            p_EP_now = 20. - (20. - p_EP) * (4. - 1. - i) / (4. - 1.)
            EP_values = torch.exp(- (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 56*56)
            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 56*56)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 56, 56)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 56, 56)

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_frequency_x = torch.roll(high_new_frequency_x,
                                              (high_new_frequency_x.shape[2] // 2, high_new_frequency_x.shape[3] // 2),
                                              dims=(2, 3))
            high_new_x_now = torch.irfft(high_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

            if i == 0:
                perturbation = self.p_perturbation0.float()  # (1, 128)
                # perturbation = (perturbation - torch.min(perturbation, 1)[0].unsqueeze(1).repeat(1, 128)) / (torch.max(perturbation, 1)[0].unsqueeze(1).repeat(1, 128) - torch.min(perturbation, 1)[0].unsqueeze(1).repeat(1, 128) + 1e-6)   # [0, 1]
                perturbation = self.sigmoid(perturbation)  # (0, 1)
                perturbation = perturbation * 19.6 + 0.2  # (0, 20-0.2)
                perturbation = torch.squeeze(perturbation).unsqueeze(1).repeat(1, 20)  # (128, 20)
                perturbation0 = perturbation
            elif i == 1:
                perturbation = self.p_perturbation1.float()  # (1, 128)
                perturbation = self.sigmoid(perturbation)  # (0, 1)
                perturbation = torch.squeeze(perturbation).unsqueeze(1).repeat(1, 20)  # (128, 20)
                perturbation = perturbation * (20. - 0.2 - perturbation0) + (
                            0.2 + perturbation0 - p_EP_now)  # (0.2 + perturbation0 - p_EP_now, 20 - p_EP_now)
                perturbation1 = perturbation
            elif i == 2:
                p_EP_last = 20. - (20. - p_EP) * (4. - 1. - 1.) / (4. - 1.)
                perturbation = self.p_perturbation2.float()  # (1, 128)
                perturbation = self.sigmoid(perturbation)  # (0, 1)
                perturbation = torch.squeeze(perturbation).unsqueeze(1).repeat(1, 20)  # (128, 20)
                perturbation = perturbation * (20. - p_EP_last - perturbation1) + (
                            p_EP_last + perturbation1 - p_EP_now)  # (p_EP_last + perturbation1 - p_EP_now, 20 - p_EP_now)
                perturbation2 = perturbation
            elif i == 3:
                p_EP_last = 20. - (20. - p_EP) * (4. - 1. - 2.) / (4. - 1.)
                perturbation = self.p_perturbation3.float()  # (1, 128)
                perturbation = self.sigmoid(perturbation)  # (0, 1)
                perturbation = torch.squeeze(perturbation).unsqueeze(1).repeat(1, 20)  # (128, 20)
                perturbation = perturbation * (20. - p_EP_last - perturbation2) + (
                            p_EP_last + perturbation2 - p_EP_now)  # (p_EP_last + perturbation2 - p_EP_now, 20 - p_EP_now)
                # perturbation2 = perturbation

            # print(torch.min(self.p_perturbation0), torch.mean(self.p_perturbation0), torch.max(self.p_perturbation0))
            # print(torch.min(perturbation), torch.mean(perturbation), torch.max(perturbation))

            p_EP_now_perturbation = p_EP_now + perturbation
            EP_values_perturbation = torch.exp(- (torch.abs(x_EP) ** p_EP_now_perturbation) / (
                        p_EP_now_perturbation * sigma_EP ** p_EP_now_perturbation + 1e-6))
            EP_values_perturbation = EP_values_perturbation.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                                                56 * 56)  # (128, 20, 56*56)
            EP_filter_high_perturbation = torch.matmul(EP_values_perturbation.permute(2, 0, 1),
                                                       all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                               0)  # (128, 1, 56*56)
            EP_filter_high_perturbation = torch.squeeze(EP_filter_high_perturbation).view(128, 56, 56)  # (128, 56, 56)
            EP_filter_high_perturbation = EP_filter_high_perturbation.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1,
                                                                                          1)  # (b, 128, 56, 56)

            # print(torch.min(EP_values), torch.mean(EP_values), torch.max(EP_values))
            # print(torch.min(EP_filter_high), torch.mean(EP_filter_high), torch.max(EP_filter_high))
            # print('--------------------------------------------')
            # print(torch.min(EP_values_perturbation), torch.mean(EP_values_perturbation), torch.max(EP_values_perturbation))
            # print(torch.min(EP_filter_high_perturbation), torch.mean(EP_filter_high_perturbation), torch.max(EP_filter_high_perturbation))

            high_new_frequency_x_perturbation = frequency_x * EP_filter_high_perturbation
            high_new_frequency_x_perturbation = torch.cat((high_new_frequency_x_perturbation.real.unsqueeze(4),
                                                           high_new_frequency_x_perturbation.imag.unsqueeze(4)), 4)
            high_new_frequency_x_perturbation = torch.roll(high_new_frequency_x_perturbation, (
            high_new_frequency_x_perturbation.shape[2] // 2, high_new_frequency_x_perturbation.shape[3] // 2),
                                                           dims=(2, 3))
            high_new_x_now_perturbation = torch.irfft(high_new_frequency_x_perturbation, 2,
                                                      onesided=False)  # (b, 128, 56, 56)

            # print(perturbation.grad, self.p_perturbation0.grad, p_EP_now.grad)

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM1_output_perturbation, Hidden_LSTM1_perturbation, Cell_LSTM1_perturbation = self.LSTM1_cell(
                    high_new_x_now_perturbation, Hidden0_LSTM1, Cell0_LSTM1)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                LSTM1_output_perturbation, Hidden_LSTM1_perturbation, Cell_LSTM1_perturbation = self.LSTM1_cell(
                    high_new_x_now_perturbation, Hidden_LSTM1_perturbation, Cell_LSTM1_perturbation)
            # ######################################

        # ############ for outputs ###############

        new_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)  # (b, 128, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        new_output_perturbation = LSTM1_output_perturbation.view(LSTM1_output_perturbation.shape[0], 128,
                                                                 56 * 56)  # (b, 128, 56*56)
        new_output_perturbation = new_output_perturbation.permute(0, 2, 1)
        new_output_perturbation = self.AddNorm1(new_output_perturbation)  # B L C
        new_output_perturbation = self.swinB.avgpool(new_output_perturbation.transpose(1, 2))  # B C 1
        new_output_perturbation = torch.flatten(new_output_perturbation, 1)
        new_output_perturbation = self.C1(new_output_perturbation)

        # ########################################

        return new_output, new_output_perturbation


class mytry5_20220901_try11_v2(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try11_v2, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.p_perturbation0 = nn.Parameter(torch.randn(1, 128).cuda())   # (1, 128)
        # self.p_perturbation0.requires_grad = True
        # self.p_perturbation1 = nn.Parameter(torch.randn(1, 128).cuda())   # (1, 128)
        # self.p_perturbation1.requires_grad = True
        # self.p_perturbation2 = nn.Parameter(torch.randn(1, 128).cuda())   # (1, 128)
        # self.p_perturbation2.requires_grad = True
        # self.p_perturbation3 = nn.Parameter(torch.randn(1, 128).cuda())   # (1, 128)
        # self.p_perturbation3.requires_grad = True

        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 112, 112)
        # #################################################################################

        # ################################# try7 20220911 #########################################
        # ori_output = x_upsample.view(x_upsample.shape[0], 128, 56 * 56)   # (b, 128, 112*112)
        # ori_output = ori_output.permute(0, 2, 1)
        # ori_output = self.AddNorm1(ori_output)  # B L C
        # ori_output = self.swinB.avgpool(ori_output.transpose(1, 2))  # B C 1
        # ori_output = torch.flatten(ori_output, 1)
        # ori_output = self.C1(ori_output)

        # print(self.p_perturbation0)
        # print(self.p_perturbation0.requires_grad)

        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        # Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        # Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1).view(1, 20, 56 * 56)  # (1, 20, 112*112)
        all_templates = all_templates.permute(1, 0, 2)  # (20, 1, 112*112)
        # print(torch.max(all_templates))

        # p_params = self.learned_ps  # (1, 128)
        # p_params = 20 * torch.rand(1, 128).cuda()  # (1, 128)
        p_params = 0.2 * torch.ones(1, 128).cuda()  # (1, 128)
        p_params = p_params.float()
        # p_params = nn.functional.softplus(p_params)  # to be positive
        # p_params = p_params.clamp(0, 20)
        sigma_EP = 20. * torch.ones(128, 20).cuda()  # (128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20] # [0,1,2,...,19]
        # x_EP = x_EP + 0.5   # [0.5, 1.5, 2.5, ..., 19.5]
        x_EP = x_EP.float()
        x_EP = x_EP.unsqueeze(0).repeat(128, 1)  # (128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(1).repeat(1, 20)  # (128, 20)

        for i in range(4):
            p_EP_now = 20. - (20. - p_EP) * (4. - 1. - i) / (4. - 1.)
            EP_values = torch.exp(- (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-6))
            EP_values = EP_values.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20,
                                                                                      56 * 56)  # (128, 20, 56*56)
            EP_filter_high = torch.matmul(EP_values.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2,
                                                                                                              0)  # (128, 1, 56*56)
            EP_filter_high = torch.squeeze(EP_filter_high).view(128, 56, 56)  # (128, 56, 56)
            EP_filter_high = EP_filter_high.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)  # (b, 128, 56, 56)

            frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])

            high_new_frequency_x = frequency_x * EP_filter_high
            high_new_frequency_x = torch.cat(
                (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
            high_new_frequency_x = torch.roll(high_new_frequency_x,
                                              (high_new_frequency_x.shape[2] // 2, high_new_frequency_x.shape[3] // 2),
                                              dims=(2, 3))
            high_new_x_now = torch.irfft(high_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

            # if i == 0:
            #     perturbation = self.p_perturbation0.float()   # (1, 128)
            #     # perturbation = (perturbation - torch.min(perturbation, 1)[0].unsqueeze(1).repeat(1, 128)) / (torch.max(perturbation, 1)[0].unsqueeze(1).repeat(1, 128) - torch.min(perturbation, 1)[0].unsqueeze(1).repeat(1, 128) + 1e-6)   # [0, 1]
            #     perturbation = self.sigmoid(perturbation)  # (0, 1)
            #     perturbation = perturbation * 19.6 + 0.2   # (0, 20-0.2)
            #     perturbation = torch.squeeze(perturbation).unsqueeze(1).repeat(1, 20)  # (128, 20)
            #     perturbation0 = perturbation
            # elif i == 1:
            #     perturbation = self.p_perturbation1.float()  # (1, 128)
            #     perturbation = self.sigmoid(perturbation)  # (0, 1)
            #     perturbation = torch.squeeze(perturbation).unsqueeze(1).repeat(1, 20)  # (128, 20)
            #     perturbation = perturbation * (20. - 0.2 - perturbation0) + (0.2 + perturbation0 - p_EP_now)  # (0.2 + perturbation0 - p_EP_now, 20 - p_EP_now)
            #     perturbation1 = perturbation
            # elif i == 2:
            #     p_EP_last = 20. - (20. - p_EP) * (4. - 1. - 1.) / (4. - 1.)
            #     perturbation = self.p_perturbation2.float()  # (1, 128)
            #     perturbation = self.sigmoid(perturbation)  # (0, 1)
            #     perturbation = torch.squeeze(perturbation).unsqueeze(1).repeat(1, 20)  # (128, 20)
            #     perturbation = perturbation * (20. - p_EP_last - perturbation1) + (p_EP_last + perturbation1 - p_EP_now)  # (p_EP_last + perturbation1 - p_EP_now, 20 - p_EP_now)
            #     perturbation2 = perturbation
            # elif i == 3:
            #     p_EP_last = 20. - (20. - p_EP) * (4. - 1. - 2.) / (4. - 1.)
            #     perturbation = self.p_perturbation3.float()  # (1, 128)
            #     perturbation = self.sigmoid(perturbation)  # (0, 1)
            #     perturbation = torch.squeeze(perturbation).unsqueeze(1).repeat(1, 20)  # (128, 20)
            #     perturbation = perturbation * (20. - p_EP_last - perturbation2) + (p_EP_last + perturbation2 - p_EP_now)  # (p_EP_last + perturbation2 - p_EP_now, 20 - p_EP_now)
            #     # perturbation2 = perturbation

            # print(torch.min(self.p_perturbation0), torch.mean(self.p_perturbation0), torch.max(self.p_perturbation0))
            # print(torch.min(perturbation), torch.mean(perturbation), torch.max(perturbation))

            # p_EP_now_perturbation = p_EP_now + perturbation
            # EP_values_perturbation = torch.exp(- (torch.abs(x_EP) ** p_EP_now_perturbation) / (p_EP_now_perturbation * sigma_EP ** p_EP_now_perturbation + 1e-6))
            # EP_values_perturbation = EP_values_perturbation.unsqueeze(2).unsqueeze(3).repeat(1, 1, 56, 56).view(128, 20, 56*56)   # (128, 20, 56*56)
            # EP_filter_high_perturbation = torch.matmul(EP_values_perturbation.permute(2, 0, 1), all_templates.permute(2, 0, 1)).permute(1, 2, 0)  # (128, 1, 56*56)
            # EP_filter_high_perturbation = torch.squeeze(EP_filter_high_perturbation).view(128, 56, 56)  # (128, 56, 56)
            # EP_filter_high_perturbation = EP_filter_high_perturbation.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1)   # (b, 128, 56, 56)

            # print(torch.min(EP_values), torch.mean(EP_values), torch.max(EP_values))
            # print(torch.min(EP_filter_high), torch.mean(EP_filter_high), torch.max(EP_filter_high))
            # print('--------------------------------------------')
            # print(torch.min(EP_values_perturbation), torch.mean(EP_values_perturbation), torch.max(EP_values_perturbation))
            # print(torch.min(EP_filter_high_perturbation), torch.mean(EP_filter_high_perturbation), torch.max(EP_filter_high_perturbation))

            # high_new_frequency_x_perturbation = frequency_x * EP_filter_high_perturbation
            # high_new_frequency_x_perturbation = torch.cat((high_new_frequency_x_perturbation.real.unsqueeze(4), high_new_frequency_x_perturbation.imag.unsqueeze(4)), 4)
            # high_new_frequency_x_perturbation = torch.roll(high_new_frequency_x_perturbation, (high_new_frequency_x_perturbation.shape[2] // 2, high_new_frequency_x_perturbation.shape[3] // 2), dims=(2, 3))
            # high_new_x_now_perturbation = torch.irfft(high_new_frequency_x_perturbation, 2, onesided=False)   # (b, 128, 56, 56)

            # print(perturbation.grad, self.p_perturbation0.grad, p_EP_now.grad)

            # ############ for LSTMs ###############
            if i == 0:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden0_LSTM1, Cell0_LSTM1)
                # LSTM1_output_perturbation, Hidden_LSTM1_perturbation, Cell_LSTM1_perturbation = self.LSTM1_cell(high_new_x_now_perturbation, Hidden0_LSTM1, Cell0_LSTM1)
            else:
                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(high_new_x_now, Hidden_LSTM1, Cell_LSTM1)
                # LSTM1_output_perturbation, Hidden_LSTM1_perturbation, Cell_LSTM1_perturbation = self.LSTM1_cell(high_new_x_now_perturbation, Hidden_LSTM1_perturbation, Cell_LSTM1_perturbation)
            # ######################################

        # ############ for outputs ###############

        new_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)  # (b, 128, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)

        # new_output_perturbation = LSTM1_output_perturbation.view(LSTM1_output_perturbation.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # new_output_perturbation = new_output_perturbation.permute(0, 2, 1)
        # new_output_perturbation = self.AddNorm1(new_output_perturbation)  # B L C
        # new_output_perturbation = self.swinB.avgpool(new_output_perturbation.transpose(1, 2))  # B C 1
        # new_output_perturbation = torch.flatten(new_output_perturbation, 1)
        # new_output_perturbation = self.C1(new_output_perturbation)

        # ########################################

        # return new_output, new_output_perturbation
        return new_output


class mytry5_20220901_try12_v1(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try12_v1, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.p_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def my_classifier(self, feature_map):
        new_output = feature_map.view(feature_map.shape[0], 128, 56 * 56)  # (b, 128, 56*56)
        # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)
        return new_output

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 56, 56)
        # #################################################################################

        # ################################# try7 20220922 #########################################
        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1)  # (1, 20, 56, 56)
        all_templates = all_templates.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1, 1)  # (b, 1, 20, 56, 56)
        all_templates = all_templates.permute(0, 3, 4, 2, 1)  # (b, 56, 56, 20, 1)

        p_params = self.p_net(x_upsample)  # (b, 128, 1, 1), instance-adaptive
        p_params = torch.squeeze(p_params)  # (b, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0.2, 20)  # [0.2, 20]
        sigma_EP = 20. * torch.ones(x_upsample.shape[0], 128, 20).cuda()  # (b, 128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.
        x_EP = x_EP.unsqueeze(0).unsqueeze(0).repeat(x_upsample.shape[0], 128, 1)  # (b, 128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(2).repeat(1, 1, 20)  # (b, 128, 20)

        # for i in range(10):
        # for i in range(4):
        # for i in range(5):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            # p_EP_now = p_EP + (20 - p_EP) * (5 - 1 - i) / (5 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # from high-pass to the full-pass
            # p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = torch.exp(
                - (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-9))  # (b, 128, 20)
            EP_values = EP_values.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 56, 56)  # (b, 128, 20, 56, 56)
            EP_values = EP_values.permute(0, 3, 4, 1, 2)  # (b, 56, 56, 128, 20)

            EP_filter_high = torch.matmul(EP_values, all_templates)  # (b, 56, 56, 128, 1)
            EP_filter_high = torch.squeeze(EP_filter_high)  # (b, 56, 56, 128)
            EP_filter_high = EP_filter_high.permute(0, 3, 1, 2)  # (b, 128, 56, 56)

            EP_filter_low = 1 - EP_filter_high  # (b, 128, 56, 56)

            # # ############ for full-to-high LSTMs ###############
            # if i == 0:
            #
            #     frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            #     frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            #     frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])   # the frequency spectrum of x_upsample
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(x_upsample, Hidden0_LSTM1, Cell0_LSTM1)
            #
            #     LSTM1_output0 = LSTM1_output
            #
            #     frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            # elif i < 5:   # i == 1, 2, 3, 4
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)
            #
            #     if i == 1:
            #         LSTM1_output1 = LSTM1_output
            #     elif i == 2:
            #         LSTM1_output2 = LSTM1_output
            #     elif i == 3:
            #         LSTM1_output3 = LSTM1_output
            #     elif i == 4:
            #         LSTM1_output4 = LSTM1_output
            #
            #     frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            #     # LSTM1_final_output = Input_LSTM1   # (b, 128, 56, 56)
            #
            # else:   # i == 5
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)   # additional step for handling p=p_learned
            #
            #     LSTM1_output5 = LSTM1_output
            #
            # # ######################################

            # ############ for high-to-full LSTMs ###############
            if i == 0:

                frequency_x = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                         dims=(2, 3))
                frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                            frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample

                frequency_x_output = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x_output = torch.roll(frequency_x_output,
                                                (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2),
                                                dims=(2, 3))
                frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])

                high_new_frequency_x = frequency_x_output * EP_filter_high

                low_new_frequency_x = frequency_x * EP_filter_low

                new_frequency_x = high_new_frequency_x + low_new_frequency_x
                # new_frequency_x = high_new_frequency_x

                new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
                new_frequency_x = torch.roll(new_frequency_x,
                                             (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2),
                                             dims=(2, 3))
                Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden0_LSTM1, Cell0_LSTM1)

                LSTM1_output0 = LSTM1_output

            # elif i < 3:   # i == 1, 2
            else:  # i == 1, 2, 3

                frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
                frequency_x_output = torch.roll(frequency_x_output,
                                                (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2),
                                                dims=(2, 3))
                frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])

                high_new_frequency_x = frequency_x_output * EP_filter_high

                low_new_frequency_x = frequency_x * EP_filter_low

                new_frequency_x = high_new_frequency_x + low_new_frequency_x

                new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
                new_frequency_x = torch.roll(new_frequency_x,
                                             (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2),
                                             dims=(2, 3))
                Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)

                if i == 1:
                    LSTM1_output1 = LSTM1_output
                elif i == 2:
                    LSTM1_output2 = LSTM1_output
                elif i == 3:
                    LSTM1_output3 = LSTM1_output
                # elif i == 4:
                #     LSTM1_output4 = LSTM1_output

                # LSTM1_final_output = Input_LSTM1   # (b, 128, 56, 56)

            # else:   # i == 5
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)   # additional step for handling p=p_learned
            #
            #     LSTM1_output5 = LSTM1_output

            # ######################################

        # ############ for outputs ###############
        # new_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)

        output0 = self.my_classifier(LSTM1_output0)
        output1 = self.my_classifier(LSTM1_output1)
        output2 = self.my_classifier(LSTM1_output2)
        output3 = self.my_classifier(LSTM1_output3)
        # output4 = self.my_classifier(LSTM1_output4)
        # output5 = self.my_classifier(LSTM1_output5)

        # ########################################

        # return new_output
        # return output0, output1, output2, output3, output4, output5
        return output0, output1, output2, output3


class mytry5_20220901_try12_v2(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try12_v2, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128)
        )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.LSTM1_add_net1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_add_net2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_add_net3 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_add_net4 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv1_kernel = nn.Parameter(torch.randn(128, 256, 3, 3).cuda())
        self.LSTM1_conv1_kernel.requires_grad = True
        self.LSTM1_conv2_kernel = nn.Parameter(torch.randn(128, 256, 3, 3).cuda())
        self.LSTM1_conv2_kernel.requires_grad = True
        self.LSTM1_conv3_kernel = nn.Parameter(torch.randn(128, 256, 3, 3).cuda())
        self.LSTM1_conv3_kernel.requires_grad = True
        self.LSTM1_conv4_kernel = nn.Parameter(torch.randn(128, 256, 3, 3).cuda())
        self.LSTM1_conv4_kernel.requires_grad = True

        # self.LSTM1_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM1_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.p_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell, x_upsample):

        LSTM1_add1 = self.LSTM1_add_net1(x_upsample)  # (b, 128, 56, 56)
        LSTM1_add2 = self.LSTM1_add_net2(x_upsample)
        LSTM1_add3 = self.LSTM1_add_net3(x_upsample)
        LSTM1_add4 = self.LSTM1_add_net4(x_upsample)

        concat_input = torch.cat((input, Hidden), 1)  # (b, 128*2, 56, 56)
        # concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 128, 7, 7)
        # concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        # concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        # concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        concat_input1 = nn.functional.conv2d(concat_input, self.LSTM1_conv1_kernel, bias=None, stride=1,
                                             padding=1)  # (b, 128, 56, 56)
        concat_input1 = concat_input1 + LSTM1_add1  # (b, 128, 56, 56)
        concat_input2 = nn.functional.conv2d(concat_input, self.LSTM1_conv2_kernel, bias=None, stride=1,
                                             padding=1)  # (b, 128, 56, 56)
        concat_input2 = concat_input2 + LSTM1_add2
        concat_input3 = nn.functional.conv2d(concat_input, self.LSTM1_conv3_kernel, bias=None, stride=1,
                                             padding=1)  # (b, 128, 56, 56)
        concat_input3 = concat_input3 + LSTM1_add3
        concat_input4 = nn.functional.conv2d(concat_input, self.LSTM1_conv4_kernel, bias=None, stride=1,
                                             padding=1)  # (b, 128, 56, 56)
        concat_input4 = concat_input4 + LSTM1_add4
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def my_classifier(self, feature_map):

        output = feature_map.view(feature_map.shape[0], 128, 56 * 56)  # (b, 128, 56*56)
        output = output.permute(0, 2, 1)
        output = self.AddNorm1(output)  # B L C
        output = self.swinB.avgpool(output.transpose(1, 2))  # B C 1
        output = torch.flatten(output, 1)
        output = self.C1(output)

        return output

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 56, 56)
        # #################################################################################

        # ################################# try7 20220922 #########################################
        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1)  # (1, 20, 56, 56)
        all_templates = all_templates.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1, 1)  # (b, 1, 20, 56, 56)
        all_templates = all_templates.permute(0, 3, 4, 2, 1)  # (b, 56, 56, 20, 1)

        p_params = self.p_net(x_upsample)  # (b, 128, 1, 1), instance-adaptive
        p_params = torch.squeeze(p_params)  # (b, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0.2, 20)  # [0.2, 20]
        sigma_EP = 20. * torch.ones(x_upsample.shape[0], 128, 20).cuda()  # (b, 128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.
        x_EP = x_EP.unsqueeze(0).unsqueeze(0).repeat(x_upsample.shape[0], 128, 1)  # (b, 128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(2).repeat(1, 1, 20)  # (b, 128, 20)

        # for i in range(10):
        # for i in range(4):
        # for i in range(5):
        # for i in range(5+1):
        # for i in range(4+1):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (
                        4 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            # p_EP_now = p_EP + (20 - p_EP) * (5 - 1 - i) / (5 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            # p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = torch.exp(
                - (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-9))  # (b, 128, 20)
            EP_values = EP_values.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 56, 56)  # (b, 128, 20, 56, 56)
            EP_values = EP_values.permute(0, 3, 4, 1, 2)  # (b, 56, 56, 128, 20)

            EP_filter_high = torch.matmul(EP_values, all_templates)  # (b, 56, 56, 128, 1)
            EP_filter_high = torch.squeeze(EP_filter_high)  # (b, 56, 56, 128)
            EP_filter_high = EP_filter_high.permute(0, 3, 1, 2)  # (b, 128, 56, 56)

            EP_filter_low = 1 - EP_filter_high  # (b, 128, 56, 56)

            # ############ for LSTMs ###############
            if i == 0:

                frequency_x = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                         dims=(2, 3))
                frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                            frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample

                frequency_x_output = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x_output = torch.roll(frequency_x_output,
                                                (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2),
                                                dims=(2, 3))
                frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])

                high_new_frequency_x = frequency_x_output * EP_filter_high

                low_new_frequency_x = frequency_x * EP_filter_low

                new_frequency_x = high_new_frequency_x + low_new_frequency_x

                new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
                new_frequency_x = torch.roll(new_frequency_x,
                                             (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2),
                                             dims=(2, 3))
                Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden0_LSTM1, Cell0_LSTM1,
                                                                         x_upsample)

                LSTM1_output0 = LSTM1_output


            # elif i < 5:   # i == 1, 2, 3, 4
            # elif i < 4:   # i == 1, 2, 3
            else:  # i == 1, 2, 3

                frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
                frequency_x_output = torch.roll(frequency_x_output,
                                                (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2),
                                                dims=(2, 3))
                frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])

                high_new_frequency_x = frequency_x_output * EP_filter_high

                low_new_frequency_x = frequency_x * EP_filter_low

                new_frequency_x = high_new_frequency_x + low_new_frequency_x

                new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
                new_frequency_x = torch.roll(new_frequency_x,
                                             (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2),
                                             dims=(2, 3))
                Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1,
                                                                         x_upsample)

                if i == 1:
                    LSTM1_output1 = LSTM1_output
                elif i == 2:
                    LSTM1_output2 = LSTM1_output
                elif i == 3:
                    LSTM1_output3 = LSTM1_output
                # elif i == 4:
                #     LSTM1_output4 = LSTM1_output

            # else:   # i == 5
            # else:   # i == 4
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1, x_upsample)   # additional step for handling p=p_learned
            #
            #     # LSTM1_output5 = LSTM1_output
            #     LSTM1_output4 = LSTM1_output

            # ######################################

        # ############ for outputs ###############
        # new_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)

        outputs0 = self.my_classifier(LSTM1_output0)
        outputs1 = self.my_classifier(LSTM1_output1)
        outputs2 = self.my_classifier(LSTM1_output2)
        outputs3 = self.my_classifier(LSTM1_output3)
        # outputs4 = self.my_classifier(LSTM1_output4)
        # outputs5 = self.my_classifier(LSTM1_output5)

        # ########################################

        # return new_output
        # return outputs0, outputs1, outputs2, outputs3, outputs4, outputs5
        # return outputs0, outputs1, outputs2, outputs3, outputs4
        return outputs0, outputs1, outputs2, outputs3


class mytry5_20220901_try12_v3(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try12_v3, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.Hidden_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.Cell_conv_block_LSTM2 = nn.Sequential(
        #     nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        # self.LSTM2_conv_block1 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block2 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block3 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )
        # self.LSTM2_conv_block4 = nn.Sequential(
        #     nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.channel_num),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
        #     # nn.BatchNorm2d(self.channel_num),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128)
        # )

        self.p_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    # def LSTM2_cell(self, input, Hidden, Cell):
    #     concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
    #     concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
    #     concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
    #     concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
    #     concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
    #     filter = self.sigmoid(concat_input1)
    #     filtered_Cell = filter * Cell
    #     enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
    #     enhanced_Cell = filtered_Cell + enhance
    #     Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
    #     Cell = enhanced_Cell
    #     output = Hidden
    #
    #     return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def my_classifier(self, feature_map):
        new_output = feature_map.view(feature_map.shape[0], 128, 56 * 56)  # (b, 128, 56*56)
        # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)
        return new_output

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 56, 56)
        # #################################################################################

        # ################################# try12_v3 20220923 #########################################
        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1)  # (1, 20, 56, 56)
        all_templates = all_templates.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1, 1)  # (b, 1, 20, 56, 56)
        all_templates = all_templates.permute(0, 3, 4, 2, 1)  # (b, 56, 56, 20, 1)

        p_params = self.p_net(x_upsample)  # (b, 128, 1, 1), instance-adaptive
        p_params = torch.squeeze(p_params)  # (b, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0.2, 20)  # [0.2, 20]
        sigma_EP = 20. * torch.ones(x_upsample.shape[0], 128, 20).cuda()  # (b, 128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.
        x_EP = x_EP.unsqueeze(0).unsqueeze(0).repeat(x_upsample.shape[0], 128, 1)  # (b, 128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(2).repeat(1, 1, 20)  # (b, 128, 20)

        # for i in range(10):
        # for i in range(4):
        # for i in range(5):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (
                        4 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            # p_EP_now = p_EP + (20 - p_EP) * (5 - 1 - i) / (5 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            # p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # from high-pass to the full-pass
            # p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = torch.exp(
                - (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-9))  # (b, 128, 20)
            EP_values = EP_values.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 56, 56)  # (b, 128, 20, 56, 56)
            EP_values = EP_values.permute(0, 3, 4, 1, 2)  # (b, 56, 56, 128, 20)

            EP_filter_high = torch.matmul(EP_values, all_templates)  # (b, 56, 56, 128, 1)
            EP_filter_high = torch.squeeze(EP_filter_high)  # (b, 56, 56, 128)
            EP_filter_high = EP_filter_high.permute(0, 3, 1, 2)  # (b, 128, 56, 56)

            EP_filter_low = 1 - EP_filter_high  # (b, 128, 56, 56)

            # ############ for full-to-high LSTMs ###############
            if i == 0:

                frequency_x = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                         dims=(2, 3))
                frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                            frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample
                high_new_frequency_x = frequency_x * EP_filter_high
                high_new_frequency_x = torch.cat(
                    (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
                high_new_frequency_x = torch.roll(high_new_frequency_x, (
                high_new_frequency_x.shape[2] // 2, high_new_frequency_x.shape[3] // 2), dims=(2, 3))
                Input_LSTM1 = torch.irfft(high_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden0_LSTM1, Cell0_LSTM1)

                # LSTM1_output0 = LSTM1_output

                # frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
                # frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
                # frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
                #
                # high_new_frequency_x = frequency_x_output * EP_filter_high
                # new_frequency_x = high_new_frequency_x
                # new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
                # new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
                # Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

            # elif i < 5:   # i == 1, 2, 3, 4
            else:  # i == 1, 2, 3

                frequency_x = torch.rfft(LSTM1_output, 2, onesided=False)
                frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                         dims=(2, 3))
                frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                            frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample
                high_new_frequency_x = frequency_x * EP_filter_high
                high_new_frequency_x = torch.cat(
                    (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
                high_new_frequency_x = torch.roll(high_new_frequency_x, (
                high_new_frequency_x.shape[2] // 2, high_new_frequency_x.shape[3] // 2), dims=(2, 3))
                Input_LSTM1 = torch.irfft(high_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)

                if i == 3:  # at the final time step
                    frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
                    frequency_x_output = torch.roll(frequency_x_output, (
                    frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
                    frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0],
                                                       frequency_x_output[:, :, :, :, 1])
                    high_new_frequency_x = frequency_x_output * EP_filter_high

                    frequency_x = torch.rfft(x_upsample, 2, onesided=False)
                    frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                             dims=(2, 3))
                    frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                                frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample
                    low_new_frequency_x = frequency_x * EP_filter_low

                    new_frequency_x = high_new_frequency_x + low_new_frequency_x
                    new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)),
                                                4)
                    new_frequency_x = torch.roll(new_frequency_x,
                                                 (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2),
                                                 dims=(2, 3))
                    final_output = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

            # if i == 1:
            #     LSTM1_output1 = LSTM1_output
            # elif i == 2:
            #     LSTM1_output2 = LSTM1_output
            # elif i == 3:
            #     LSTM1_output3 = LSTM1_output
            # elif i == 4:
            #     LSTM1_output4 = LSTM1_output

            #     frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            #     # LSTM1_final_output = Input_LSTM1   # (b, 128, 56, 56)
            #
            # else:   # i == 5
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)   # additional step for handling p=p_learned
            #
            #     LSTM1_output5 = LSTM1_output

            # ######################################

            # ############ for high-to-full LSTMs ###############
            # if i == 0:
            #
            #     frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            #     frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            #     frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])   # the frequency spectrum of x_upsample
            #
            #     frequency_x_output = torch.rfft(x_upsample, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #     # new_frequency_x = high_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden0_LSTM1, Cell0_LSTM1)
            #
            #     LSTM1_output0 = LSTM1_output
            #
            # # elif i < 3:   # i == 1, 2
            # else:   # i == 1, 2, 3
            #
            #     frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)
            #
            #     if i == 1:
            #         LSTM1_output1 = LSTM1_output
            #     elif i == 2:
            #         LSTM1_output2 = LSTM1_output
            #     elif i == 3:
            #         LSTM1_output3 = LSTM1_output
            #     # elif i == 4:
            #     #     LSTM1_output4 = LSTM1_output
            #
            #     # LSTM1_final_output = Input_LSTM1   # (b, 128, 56, 56)
            #
            # # else:   # i == 5
            # #
            # #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)   # additional step for handling p=p_learned
            # #
            # #     LSTM1_output5 = LSTM1_output

            # ######################################

        # ############ for outputs ###############
        # new_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)

        # output0 = self.my_classifier(LSTM1_output0)
        # output1 = self.my_classifier(LSTM1_output1)
        # output2 = self.my_classifier(LSTM1_output2)
        # output3 = self.my_classifier(LSTM1_output3)
        # output4 = self.my_classifier(LSTM1_output4)
        # output5 = self.my_classifier(LSTM1_output5)

        new_output = self.my_classifier(final_output)

        # ########################################

        return new_output
        # return output0, output1, output2, output3, output4, output5
        # return output0, output1, output2, output3


class mytry5_20220901_try12_v4(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try12_v4, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.p_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def my_classifier(self, feature_map):
        new_output = feature_map.view(feature_map.shape[0], 128, 56 * 56)  # (b, 128, 56*56)
        # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)
        return new_output

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 56, 56)
        # #################################################################################

        # ################################# try12_v4 20220924 #########################################
        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1)  # (1, 20, 56, 56)
        all_templates = all_templates.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1, 1)  # (b, 1, 20, 56, 56)
        all_templates = all_templates.permute(0, 3, 4, 2, 1)  # (b, 56, 56, 20, 1)

        p_params = self.p_net(x_upsample)  # (b, 128, 1, 1), instance-adaptive
        p_params = torch.squeeze(p_params)  # (b, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0.2, 20)  # [0.2, 20]
        sigma_EP = 20. * torch.ones(x_upsample.shape[0], 128, 20).cuda()  # (b, 128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.
        x_EP = x_EP.unsqueeze(0).unsqueeze(0).repeat(x_upsample.shape[0], 128, 1)  # (b, 128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(2).repeat(1, 1, 20)  # (b, 128, 20)

        # for i in range(10):
        # for i in range(4):
        # for i in range(5):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (
                        4 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            # p_EP_now = p_EP + (20 - p_EP) * (5 - 1 - i) / (5 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            # p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # from high-pass to the full-pass
            # p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = torch.exp(
                - (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-9))  # (b, 128, 20)
            EP_values = EP_values.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 56, 56)  # (b, 128, 20, 56, 56)
            EP_values = EP_values.permute(0, 3, 4, 1, 2)  # (b, 56, 56, 128, 20)

            EP_filter_high = torch.matmul(EP_values, all_templates)  # (b, 56, 56, 128, 1)
            EP_filter_high = torch.squeeze(EP_filter_high)  # (b, 56, 56, 128)
            EP_filter_high = EP_filter_high.permute(0, 3, 1, 2)  # (b, 128, 56, 56)

            EP_filter_low = 1 - EP_filter_high  # (b, 128, 56, 56)

            # ############ for full-to-high LSTMs ###############
            if i == 0:

                frequency_x = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                         dims=(2, 3))
                frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                            frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample
                high_new_frequency_x = frequency_x * EP_filter_high
                high_new_frequency_x = torch.cat(
                    (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
                high_new_frequency_x = torch.roll(high_new_frequency_x, (
                high_new_frequency_x.shape[2] // 2, high_new_frequency_x.shape[3] // 2), dims=(2, 3))
                Input_LSTM1 = torch.irfft(high_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                low_new_frequency_x = frequency_x * EP_filter_low
                low_new_frequency_x = torch.cat(
                    (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
                low_new_frequency_x = torch.roll(low_new_frequency_x,
                                                 (low_new_frequency_x.shape[2] // 2, low_new_frequency_x.shape[3] // 2),
                                                 dims=(2, 3))
                Input_LSTM2 = torch.irfft(low_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM1_cell(Input_LSTM2, Hidden0_LSTM2, Cell0_LSTM2)

                # LSTM1_output0 = LSTM1_output

                # frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
                # frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
                # frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
                #
                # high_new_frequency_x = frequency_x_output * EP_filter_high
                # new_frequency_x = high_new_frequency_x
                # new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
                # new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
                # Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

            # elif i < 5:   # i == 1, 2, 3, 4
            else:  # i == 1, 2, 3

                frequency_x = torch.rfft(LSTM1_output, 2, onesided=False)
                frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                         dims=(2, 3))
                frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                            frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample
                high_new_frequency_x = frequency_x * EP_filter_high
                high_new_frequency_x = torch.cat(
                    (high_new_frequency_x.real.unsqueeze(4), high_new_frequency_x.imag.unsqueeze(4)), 4)
                high_new_frequency_x = torch.roll(high_new_frequency_x, (
                high_new_frequency_x.shape[2] // 2, high_new_frequency_x.shape[3] // 2), dims=(2, 3))
                Input_LSTM1 = torch.irfft(high_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                low_new_frequency_x = frequency_x * EP_filter_low
                low_new_frequency_x = torch.cat(
                    (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
                low_new_frequency_x = torch.roll(low_new_frequency_x,
                                                 (low_new_frequency_x.shape[2] // 2, low_new_frequency_x.shape[3] // 2),
                                                 dims=(2, 3))
                Input_LSTM2 = torch.irfft(low_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)

                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM1_cell(Input_LSTM2, Hidden0_LSTM2, Cell0_LSTM2)

                if i == 3:  # at the final time step
                    frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
                    frequency_x_output = torch.roll(frequency_x_output, (
                    frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
                    frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0],
                                                       frequency_x_output[:, :, :, :, 1])
                    high_new_frequency_x = frequency_x_output * EP_filter_high

                    # frequency_x = torch.rfft(x_upsample, 2, onesided=False)
                    frequency_x = torch.rfft(LSTM2_output, 2, onesided=False)
                    frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                             dims=(2, 3))
                    frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                                frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample
                    low_new_frequency_x = frequency_x * EP_filter_low

                    new_frequency_x = high_new_frequency_x + low_new_frequency_x
                    new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)),
                                                4)
                    new_frequency_x = torch.roll(new_frequency_x,
                                                 (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2),
                                                 dims=(2, 3))
                    final_output = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

            # if i == 1:
            #     LSTM1_output1 = LSTM1_output
            # elif i == 2:
            #     LSTM1_output2 = LSTM1_output
            # elif i == 3:
            #     LSTM1_output3 = LSTM1_output
            # elif i == 4:
            #     LSTM1_output4 = LSTM1_output

            #     frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            #     # LSTM1_final_output = Input_LSTM1   # (b, 128, 56, 56)
            #
            # else:   # i == 5
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)   # additional step for handling p=p_learned
            #
            #     LSTM1_output5 = LSTM1_output

            # ######################################

            # ############ for high-to-full LSTMs ###############
            # if i == 0:
            #
            #     frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            #     frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            #     frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])   # the frequency spectrum of x_upsample
            #
            #     frequency_x_output = torch.rfft(x_upsample, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #     # new_frequency_x = high_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden0_LSTM1, Cell0_LSTM1)
            #
            #     LSTM1_output0 = LSTM1_output
            #
            # # elif i < 3:   # i == 1, 2
            # else:   # i == 1, 2, 3
            #
            #     frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)
            #
            #     if i == 1:
            #         LSTM1_output1 = LSTM1_output
            #     elif i == 2:
            #         LSTM1_output2 = LSTM1_output
            #     elif i == 3:
            #         LSTM1_output3 = LSTM1_output
            #     # elif i == 4:
            #     #     LSTM1_output4 = LSTM1_output
            #
            #     # LSTM1_final_output = Input_LSTM1   # (b, 128, 56, 56)
            #
            # # else:   # i == 5
            # #
            # #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)   # additional step for handling p=p_learned
            # #
            # #     LSTM1_output5 = LSTM1_output

            # ######################################

        # ############ for outputs ###############
        # new_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)

        # output0 = self.my_classifier(LSTM1_output0)
        # output1 = self.my_classifier(LSTM1_output1)
        # output2 = self.my_classifier(LSTM1_output2)
        # output3 = self.my_classifier(LSTM1_output3)
        # output4 = self.my_classifier(LSTM1_output4)
        # output5 = self.my_classifier(LSTM1_output5)

        new_output = self.my_classifier(final_output)

        # ########################################

        return new_output
        # return output0, output1, output2, output3, output4, output5
        # return output0, output1, output2, output3


class mytry5_20220901_try12_v5(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(mytry5_20220901_try12_v5, self).__init__()
        self.swinB = transformer
        # self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.avgpool14 = nn.AdaptiveAvgPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        # self.maxpool14 = nn.AdaptiveMaxPool2d((14, 14))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.C1 = nn.Linear(128, num_classes)
        # self.C2 = nn.Linear(128, num_classes)
        # self.C3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes
        self.channel_num = 64

        self.relu = nn.ReLU()

        self.Hidden_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM1 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Hidden_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.Cell_conv_block_LSTM2 = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.LSTM1_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM1_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block1 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block2 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block3 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.LSTM2_conv_block4 = nn.Sequential(
            nn.Conv2d(2 * 128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.p_net = nn.Sequential(
            nn.Conv2d(128, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_num, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.convert_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,  # 输入数据的通道数
                               out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(self.channel_num),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
            #                    out_channels=self.channel_num,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
            #                    kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
            #                    stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
            #                    padding=1,  # 原图周围需要填充的格子行（列）数
            #                    output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
            #                    groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
            #                    bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
            #                    ),
            # nn.BatchNorm2d(self.channel_num),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.channel_num,  # 输入数据的通道数
                               out_channels=128,  # 输出数据的通道数（就是我想让输出多少通道，就设置为多少）
                               kernel_size=4,  # 卷积核的尺寸（如（3，2），3与（3,3）等同）
                               stride=2,  # 卷积步长，就是卷积操作时每次移动的格子数
                               padding=1,  # 原图周围需要填充的格子行（列）数
                               output_padding=0,  # 输出特征图边缘需要填充的行（列）数，一般不设置
                               groups=1,  # 分组卷积的组数，一般默认设置为1，不用管
                               bias=False  # 卷积偏置，一般设置为False，True的话可以增加模型的泛化能力
                               ),
            nn.BatchNorm2d(128)
            # nn.ReLU(inplace=True)
        )

        self.AddNorm1 = nn.LayerNorm(128)
        # self.AddNorm2 = nn.LayerNorm(128)
        # self.AddNorm3 = nn.LayerNorm(128)

        # self.learned_ps = nn.Parameter(torch.ones(1, 128).cuda(), requires_grad=True)   # (1, 128)
        self.templates = self.generate_templates(num_templates=20)

    def LSTM1_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM1_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM1_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM1_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM1_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def LSTM2_cell(self, input, Hidden, Cell):
        concat_input = torch.cat((input, Hidden), 1)  # (b, 1024*2, 7, 7)
        concat_input1 = self.LSTM2_conv_block1(concat_input)  # (b, 1024, 7, 7)
        concat_input2 = self.LSTM2_conv_block2(concat_input)  # (b, 1024, 7, 7)
        concat_input3 = self.LSTM2_conv_block3(concat_input)  # (b, 1024, 7, 7)
        concat_input4 = self.LSTM2_conv_block4(concat_input)  # (b, 1024, 7, 7)
        filter = self.sigmoid(concat_input1)
        filtered_Cell = filter * Cell
        enhance = self.tanh(concat_input2) * self.sigmoid(concat_input3)
        enhanced_Cell = filtered_Cell + enhance
        Hidden = self.tanh(Cell) * self.sigmoid(concat_input4)
        Cell = enhanced_Cell
        output = Hidden

        return output, Hidden, Cell

    def generate_templates(self, num_templates=20):
        # templates = torch.zeros(num_templates, 56, 56).cuda()   # (20, 112, 112)
        templates = torch.ones(num_templates, 56, 56).cuda()  # (20, 112, 112)
        sides_list = [2 * int(28 / num_templates * j) for j in range(num_templates)]
        sides_list.append(56)
        for i in range(num_templates):
            side_larger = sides_list[i + 1]
            side_smaller = sides_list[i]
            padding_side_smaller = int((56 - side_smaller) / 2)
            padding_side_larger = int((56 - side_larger) / 2)
            pad_layer_smaller = nn.ConstantPad2d(
                padding=(padding_side_smaller, padding_side_smaller, padding_side_smaller, padding_side_smaller),
                value=1.)
            pad_layer_larger = nn.ZeroPad2d(
                padding=(padding_side_larger, padding_side_larger, padding_side_larger, padding_side_larger))
            high_mask_smaller = torch.zeros(side_smaller, side_smaller).cuda()
            high_mask_smaller = pad_layer_smaller(high_mask_smaller)
            high_mask_larger = torch.ones(side_larger, side_larger).cuda()
            high_mask_larger = pad_layer_larger(high_mask_larger)
            templates[i, :, :] = templates[i, :, :] * high_mask_smaller * high_mask_larger  # (20, 56, 56)

        return templates

    def my_classifier(self, feature_map):
        new_output = feature_map.view(feature_map.shape[0], 128, 56 * 56)  # (b, 128, 56*56)
        # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        new_output = new_output.permute(0, 2, 1)
        new_output = self.AddNorm1(new_output)  # B L C
        new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        new_output = torch.flatten(new_output, 1)
        new_output = self.C1(new_output)
        return new_output

    def forward(self, x):
        x = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        # x0 = x.permute(0, 2, 1)
        # x0 = x0.view(x0.shape[0], 128, 112, 112)
        layer_num = 0
        for layer in self.swinB.layers:
            x = layer(x)
            layer_num += 1
            # if layer_num == 1:
            #     x1 = x.permute(0, 2, 1)
            #     x1 = x1.view(x1.shape[0], 256, 56, 56)
            # elif layer_num == 2:
            #     x2 = x.permute(0, 2, 1)
            #     x2 = x2.view(x2.shape[0], 512, 28, 28)
            # elif layer_num == 3:
            #     x3 = x.permute(0, 2, 1)
            #     x3 = x3.view(x3.shape[0], 1024, 14, 14)
            # elif layer_num == 4:
            if layer_num == 4:
                x4 = x.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], 1024, 14, 14)

        # ############################# Upsampling 20220911 ###############################
        x_upsample = self.convert_conv_block(x4)  # from (b, 1024, 14, 14) to (b, 128, 56, 56)
        # #################################################################################

        # ################################# try12_v5 20220924 #########################################
        Hidden0_LSTM1 = self.Hidden_conv_block_LSTM1(x_upsample)
        Cell0_LSTM1 = self.Cell_conv_block_LSTM1(x_upsample)
        Hidden0_LSTM2 = self.Hidden_conv_block_LSTM2(x_upsample)
        Cell0_LSTM2 = self.Cell_conv_block_LSTM2(x_upsample)

        all_templates = self.templates.unsqueeze(0).repeat(1, 1, 1, 1)  # (1, 20, 56, 56)
        all_templates = all_templates.unsqueeze(0).repeat(x_upsample.shape[0], 1, 1, 1, 1)  # (b, 1, 20, 56, 56)
        all_templates = all_templates.permute(0, 3, 4, 2, 1)  # (b, 56, 56, 20, 1)

        p_params = self.p_net(x_upsample)  # (b, 128, 1, 1), instance-adaptive
        p_params = torch.squeeze(p_params)  # (b, 128)
        p_params = nn.functional.softplus(p_params)  # to be positive
        p_params = p_params.clamp(0.2, 20)  # [0.2, 20]
        sigma_EP = 20. * torch.ones(x_upsample.shape[0], 128, 20).cuda()  # (b, 128, 20)
        x_EP = torch.tensor(list(range(20))).cuda()  # [20]
        x_EP = x_EP + 0.
        x_EP = x_EP.unsqueeze(0).unsqueeze(0).repeat(x_upsample.shape[0], 128, 1)  # (b, 128, 20)
        p_EP = torch.squeeze(p_params).unsqueeze(2).repeat(1, 1, 20)  # (b, 128, 20)

        # for i in range(10):
        # for i in range(4):
        # for i in range(5):
        for i in range(4):
            # print(i)
            # p_EP_now = p_EP + (20 - p_EP) * (10 - 1 - i) / (10 - 1)
            # p_EP_now = p_EP + (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            # p_EP_now = p_EP + (20 - p_EP) * (5 - 1 - i) / (5 - 1)  # from full-pass (set p to 20) to the high-pass with the instance-adaptive learned ps
            p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)  # from high-pass to the full-pass
            # p_EP_now = 20 - (20 - p_EP) * (4 - 1 - i) / (4 - 1)
            EP_values = torch.exp(
                - (torch.abs(x_EP) ** p_EP_now) / (p_EP_now * sigma_EP ** p_EP_now + 1e-9))  # (b, 128, 20)
            EP_values = EP_values.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 56, 56)  # (b, 128, 20, 56, 56)
            EP_values = EP_values.permute(0, 3, 4, 1, 2)  # (b, 56, 56, 128, 20)

            EP_filter_high = torch.matmul(EP_values, all_templates)  # (b, 56, 56, 128, 1)
            EP_filter_high = torch.squeeze(EP_filter_high)  # (b, 56, 56, 128)
            EP_filter_high = EP_filter_high.permute(0, 3, 1, 2)  # (b, 128, 56, 56)

            EP_filter_low = 1 - EP_filter_high  # (b, 128, 56, 56)

            # # ############ for full-to-high LSTMs ###############
            # if i == 0:
            #
            #     frequency_x = torch.rfft(x_upsample, 2, onesided=False)
            #     frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2), dims=(2, 3))
            #     frequency_x = torch.complex(frequency_x[:, :, :, :, 0], frequency_x[:, :, :, :, 1])   # the frequency spectrum of x_upsample
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(x_upsample, Hidden0_LSTM1, Cell0_LSTM1)
            #
            #     LSTM1_output0 = LSTM1_output
            #
            #     frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            # elif i < 5:   # i == 1, 2, 3, 4
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)
            #
            #     if i == 1:
            #         LSTM1_output1 = LSTM1_output
            #     elif i == 2:
            #         LSTM1_output2 = LSTM1_output
            #     elif i == 3:
            #         LSTM1_output3 = LSTM1_output
            #     elif i == 4:
            #         LSTM1_output4 = LSTM1_output
            #
            #     frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
            #     frequency_x_output = torch.roll(frequency_x_output, (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2), dims=(2, 3))
            #     frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])
            #
            #     high_new_frequency_x = frequency_x_output * EP_filter_high
            #
            #     low_new_frequency_x = frequency_x * EP_filter_low
            #
            #     new_frequency_x = high_new_frequency_x + low_new_frequency_x
            #
            #     new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
            #     new_frequency_x = torch.roll(new_frequency_x, (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2), dims=(2, 3))
            #     Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)
            #
            #     # LSTM1_final_output = Input_LSTM1   # (b, 128, 56, 56)
            #
            # else:   # i == 5
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)   # additional step for handling p=p_learned
            #
            #     LSTM1_output5 = LSTM1_output
            #
            # # ######################################

            # ############ for high-to-full LSTMs ###############
            if i == 0:

                frequency_x = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x = torch.roll(frequency_x, (frequency_x.shape[2] // 2, frequency_x.shape[3] // 2),
                                         dims=(2, 3))
                frequency_x = torch.complex(frequency_x[:, :, :, :, 0],
                                            frequency_x[:, :, :, :, 1])  # the frequency spectrum of x_upsample

                frequency_x_output = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x_output = torch.roll(frequency_x_output,
                                                (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2),
                                                dims=(2, 3))
                frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])

                high_new_frequency_x = frequency_x_output * EP_filter_high

                low_new_frequency_x = frequency_x * EP_filter_low

                new_frequency_x = high_new_frequency_x + low_new_frequency_x
                # new_frequency_x = high_new_frequency_x

                new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
                new_frequency_x = torch.roll(new_frequency_x,
                                             (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2),
                                             dims=(2, 3))
                Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                # Input_LSTM2 = Input_LSTM1

                low_new_frequency_x = torch.cat(
                    (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
                low_new_frequency_x = torch.roll(low_new_frequency_x,
                                                 (low_new_frequency_x.shape[2] // 2, low_new_frequency_x.shape[3] // 2),
                                                 dims=(2, 3))
                Input_LSTM2 = torch.irfft(low_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden0_LSTM1, Cell0_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(Input_LSTM2, Hidden0_LSTM2, Cell0_LSTM2)

                # LSTM1_output0 = LSTM1_output

            # elif i < 3:   # i == 1, 2
            else:  # i == 1, 2, 3

                frequency_x_output = torch.rfft(LSTM1_output, 2, onesided=False)
                frequency_x_output = torch.roll(frequency_x_output,
                                                (frequency_x_output.shape[2] // 2, frequency_x_output.shape[3] // 2),
                                                dims=(2, 3))
                frequency_x_output = torch.complex(frequency_x_output[:, :, :, :, 0], frequency_x_output[:, :, :, :, 1])

                frequency_x_output2 = torch.rfft(x_upsample, 2, onesided=False)
                frequency_x_output2 = torch.roll(frequency_x_output2,
                                                 (frequency_x_output2.shape[2] // 2, frequency_x_output2.shape[3] // 2),
                                                 dims=(2, 3))
                frequency_x_output2 = torch.complex(frequency_x_output2[:, :, :, :, 0],
                                                    frequency_x_output2[:, :, :, :, 1])

                high_new_frequency_x = frequency_x_output * EP_filter_high

                # low_new_frequency_x = frequency_x * EP_filter_low
                low_new_frequency_x = frequency_x_output2 * EP_filter_low

                new_frequency_x = high_new_frequency_x + low_new_frequency_x

                new_frequency_x = torch.cat((new_frequency_x.real.unsqueeze(4), new_frequency_x.imag.unsqueeze(4)), 4)
                new_frequency_x = torch.roll(new_frequency_x,
                                             (new_frequency_x.shape[2] // 2, new_frequency_x.shape[3] // 2),
                                             dims=(2, 3))
                Input_LSTM1 = torch.irfft(new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                low_new_frequency_x = torch.cat(
                    (low_new_frequency_x.real.unsqueeze(4), low_new_frequency_x.imag.unsqueeze(4)), 4)
                low_new_frequency_x = torch.roll(low_new_frequency_x,
                                                 (low_new_frequency_x.shape[2] // 2, low_new_frequency_x.shape[3] // 2),
                                                 dims=(2, 3))
                Input_LSTM2 = torch.irfft(low_new_frequency_x, 2, onesided=False)  # (b, 128, 56, 56)

                LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)
                LSTM2_output, Hidden_LSTM2, Cell_LSTM2 = self.LSTM2_cell(Input_LSTM2, Hidden_LSTM2, Cell_LSTM2)

                # if i == 1:
                #     LSTM1_output1 = LSTM1_output
                # elif i == 2:
                #     LSTM1_output2 = LSTM1_output
                # elif i == 3:
                #     LSTM1_output3 = LSTM1_output
                # elif i == 4:
                #     LSTM1_output4 = LSTM1_output

                # LSTM1_final_output = Input_LSTM1   # (b, 128, 56, 56)

            # else:   # i == 5
            #
            #     LSTM1_output, Hidden_LSTM1, Cell_LSTM1 = self.LSTM1_cell(Input_LSTM1, Hidden_LSTM1, Cell_LSTM1)   # additional step for handling p=p_learned
            #
            #     LSTM1_output5 = LSTM1_output

            # ######################################

        # ############ for outputs ###############
        # new_output = LSTM1_output.view(LSTM1_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # # new_output = LSTM1_final_output.view(LSTM1_final_output.shape[0], 128, 56 * 56)   # (b, 128, 56*56)
        # new_output = new_output.permute(0, 2, 1)
        # new_output = self.AddNorm1(new_output)  # B L C
        # new_output = self.swinB.avgpool(new_output.transpose(1, 2))  # B C 1
        # new_output = torch.flatten(new_output, 1)
        # new_output = self.C1(new_output)

        # output0 = self.my_classifier(LSTM1_output0)
        # output1 = self.my_classifier(LSTM1_output1)
        # output2 = self.my_classifier(LSTM1_output2)
        # output3 = self.my_classifier(LSTM1_output3)
        output = self.my_classifier(LSTM1_output)
        # output4 = self.my_classifier(LSTM1_output4)
        # output5 = self.my_classifier(LSTM1_output5)

        # ########################################

        # return new_output
        # return output0, output1, output2, output3, output4, output5
        # return output0, output1, output2, output3
        return output


# ################################################################################################
# ############################################ main ##############################################
# ################################################################################################
# Train params
save_dir = 'open_set_recognition/methods/baseline/ensemble_entropy_test'    # Evaluation save dir
args = parser.parse_args()
args.save_dir = save_dir
args.use_supervised_places = False

device = torch.device('cuda:0')

root_model_path = 'open_set_recognition/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = 'open_set_recognition/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'
osr_split_dir = 'data/open_set_splits'

all_preds = []

# Get OSR splits
osr_path = os.path.join(osr_split_dir, '{}_osr_splits.pkl'.format(args.dataset))

with open(osr_path, 'rb') as f:
    class_info = pickle.load(f)

train_classes = class_info['known_classes']
open_set_classes = class_info['unknown_classes']


def my_cross_entropy_loss(input, target):
    # input.shape: torch.size([-1, class])
    # target.shape: torch.size([-1])
    exp = torch.exp(input)
    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
    tmp2 = exp.sum(1)
    softmax = tmp1 / tmp2
    log = -torch.log(softmax)

    return log


for difficulty in ('Easy', 'Medium', 'Hard'):

    # ------------------------
    # DATASETS
    # ------------------------

    args.train_classes, args.open_set_classes = train_classes, open_set_classes[difficulty]

    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            image_size=args.image_size, balance_open_set_eval=False,
                            split_train_val=False, open_set_classes=args.open_set_classes)

    # for i in range(0, len(datasets['train']), 10):
        # print(i, list(datasets['train'].data.filepath)[i])

    # ------------------------
    # DATALOADERS
    # ------------------------
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = False
        # shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                    shuffle=shuffle, sampler=None, num_workers=args.num_workers)

    # #######################################################################################
    # ############### for saving training data as .npy format ################################
    # #######################################################################################
    # all_data_train = []
    # all_data_train_labels = []
    # for data, labels, idx in tqdm(dataloaders['train']):
    #     all_data_train += data.tolist()
    #     all_data_train_labels += labels.tolist()
    # all_data_train = np.asarray(all_data_train)
    # all_data_train_labels = np.asarray(all_data_train_labels)
    # np.save('all_data_train_cub.npy', all_data_train)
    # np.save('all_data_train_labels_cub.npy', all_data_train_labels)
    #
    # import sys
    # sys.exit(0)
    ######################## for saving test-known npy data #################################
    # all_data_test_known = []
    # all_data_test_known_labels = []
    # for data, labels, idx in tqdm(dataloaders['test_known']):
    #     all_data_test_known += data.tolist()
    #     all_data_test_known_labels += labels.tolist()
    #     with torch.no_grad():
    #         outputs = net(data)
    #         predictions = outputs_ori.data.max(1)[1]
    #
    # all_data_test_known = np.asarray(all_data_test_known)
    # all_data_test_known_labels = np.asarray(all_data_test_known_labels)
    # np.save('all_data_test_known_cub.npy', all_data_test_known)
    # np.save('all_data_test_known_labels_cub.npy', all_data_test_known_labels)
    #
    # import sys
    # sys.exit(0)
    # #######################################################################################
    # #######################################################################################
    # #######################################################################################

    # ------------------------
    # MODEL
    # ------------------------
    print('Loading model...')

    F = SwinTransformer(img_size=448, patch_size=4, in_chans=3, num_classes=1000,
                        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        # window_size=14, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False
                        # use_checkpoint=True
                        )  # the feature dim is 1024

    # net = mytry5_20220512_v3_1(F, num_classes=len(args.train_classes))
    # net = mytry5_20220512(F, num_classes=len(args.train_classes))

    # net = mytry5_20220901_try6(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v1(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v2(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v1_p1(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v1_p2(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v1_p3(F, num_classes=len(args.train_classes))
    net = mytry5_20220901_try7_v1_p4(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v1_p5(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v1_p6(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v1_p7(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try7_v1_p8(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try8_v1(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try8_v2(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try8_v3(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try11_v1(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try11_v2(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try12_v1(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try12_v2(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try12_v3(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try12_v4(F, num_classes=len(args.train_classes))
    # net = mytry5_20220901_try12_v5(F, num_classes=len(args.train_classes))

    # net = openSetClassifier(len(args.train_classes), 3, 448)

    # net.load_state_dict(torch.load('open_set_recognition/log/(13.05.2022_|_30.604)/arpl_models/cub/checkpoints/cub_100net1_Softmax.pth'))   # mytry5_20220512_swinB_v3_1
    # net.load_state_dict(torch.load('open_set_recognition/log/(31.05.2022_|_25.861)/arpl_models/aircraft/checkpoints/aircraft_100net1_Softmax.pth'))   # aircraft backbone

    # net.load_state_dict(torch.load('open_set_recognition/log/(08.09.2022_|_52.356)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try6_CUB_Aircraft cub
    # net.load_state_dict(torch.load('open_set_recognition/log/(08.09.2022_|_03.533)/arpl_models/aircraft/checkpoints/aircraft_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try6_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(09.09.2022_|_51.841)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try6_CUB_Aircraft cub
    # net.load_state_dict(torch.load('open_set_recognition/log/(11.09.2022_|_24.436)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_CUB_Aircraft cub
    # net.load_state_dict(torch.load('open_set_recognition/log/(11.09.2022_|_54.237)/arpl_models/aircraft/checkpoints/aircraft_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(12.09.2022_|_25.059)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_CUB_Aircraft cub
    # net.load_state_dict(torch.load('open_set_recognition/log/(13.09.2022_|_29.231)/arpl_models/aircraft/checkpoints/aircraft_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(13.09.2022_|_32.469)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v2_CUB_Aircraft cub
    # net.load_state_dict(torch.load('open_set_recognition/log/(13.09.2022_|_36.067)/arpl_models/aircraft/checkpoints/aircraft_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v2_CUB_Aircraft aircraft

    # net.load_state_dict(torch.load('open_set_recognition/log/(14.09.2022_|_44.552)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p1_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(14.09.2022_|_37.952)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p2_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(14.09.2022_|_40.308)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p3_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(15.09.2022_|_06.705)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p4_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(15.09.2022_|_31.505)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p5_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(15.09.2022_|_58.220)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p6_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(16.09.2022_|_16.528)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p7_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(16.09.2022_|_43.761)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p8_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(18.09.2022_|_03.143)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try8_v1_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(18.09.2022_|_09.949)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try8_v2_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(18.09.2022_|_24.498)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try8_v3_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(21.09.2022_|_37.120)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try11_v1_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(22.09.2022_|_48.728)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try11_v2_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(23.09.2022_|_51.828)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try12_v1_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(23.09.2022_|_58.797)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try12_v2_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(23.09.2022_|_04.718)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try12_v3_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(24.09.2022_|_13.854)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try12_v4_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(24.09.2022_|_26.303)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try12_v5_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(26.09.2022_|_59.054)/arpl_models/aircraft/checkpoints/aircraft_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p4_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(28.09.2022_|_40.582)/arpl_models/aircraft/checkpoints/aircraft_bestaurocnet1_Softmax.pth'))   # ** mytry20220905_swinB_try7_v1_p4_CUB_Aircraft aircraft

    # net.load_state_dict(torch.load('open_set_recognition/log/(21.09.2022_|_43.171)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p2_new_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(21.09.2022_|_43.754)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p3_new_CUB_Aircraft aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(21.09.2022_|_57.569)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # mytry20220905_swinB_try7_v1_p4_new_CUB_Aircraft aircraft
    net.load_state_dict(torch.load('open_set_recognition/log/(22.09.2022_|_26.413)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # ** mytry20220905_swinB_try7_v1_p4_new_CUB_Aircraft aircraft

    # net.load_state_dict(torch.load('open_set_recognition/log/(21.09.2022_|_56.456)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # for MoEP-AE cub
    # net.load_state_dict(torch.load('open_set_recognition/log/(22.09.2022_|_35.642)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # for MoEP-AE cub another try
    # net.load_state_dict(torch.load('open_set_recognition/log/(22.09.2022_|_51.296)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # for MoEP-AE cub another try
    # net.load_state_dict(torch.load('open_set_recognition/log/(23.09.2022_|_44.808)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # for MoEP-AE cub another try
    # net.load_state_dict(torch.load('open_set_recognition/log/(23.09.2022_|_50.937)/arpl_models/cub/checkpoints/cub_bestaurocnet1_Softmax.pth'))   # for MoEP-AE cub another try
    # net.load_state_dict(torch.load('open_set_recognition/log/(21.09.2022_|_35.852)/arpl_models/aircraft/checkpoints/aircraft_bestaurocnet1_Softmax.pth'))   # for MoEP-AE aircraft
    # net.load_state_dict(torch.load('open_set_recognition/log/(22.09.2022_|_58.507)/arpl_models/aircraft/checkpoints/aircraft_bestaurocnet1_Softmax.pth'))   # for MoEP-AE aircraft another try

    net = net.to(device)
    net.eval()

    correct, total = 0, 0
    correct2, total2 = 0, 0
    CEloss = nn.CrossEntropyLoss()

    torch.cuda.empty_cache()

    _pred_k_acc, _pred_k, _pred_u, _labels = [], [], [], []
    _pred_k_acc2, _pred_k2, _pred_u2 = [], [], []

    # with torch.no_grad():
    # if difficulty == 'Hard':
    #     all_data_test_known = []

    # if difficulty == 'Easy':
    #     all_test_known_logits1 = []
    #     all_test_known_logits2 = []
    #     all_test_known_logits3 = []
    #     all_test_known_logits4 = []
    #     all_test_known_logits5 = []

    batch_num = 0
    for data, labels, idx in tqdm(dataloaders['test_known']):
        data, labels = data.cuda(), labels.cuda()

        # if difficulty == 'Hard':
        #     all_data_test_known += data.detach().cpu().tolist()

        with torch.no_grad():
            # outputs = net(data)
            # outputs_ori, outputs_LSTM, attention_map = net(data)
            # outputs_ori = net(data)
            # ori_output, low_output, high_output = net(data)
            # outputs_ori = net(data)
            # outputs_ori, outputs_LSTM = net(data)
            outputs_ori = net(data)
            # outputs_ori, _, _, _, _ = net(data, None)
            # _, _, _, outputs_ori = net(data)
            # outputs_ori, _ = net(data)
            outputs_LSTM = outputs_ori
            # outputs_ori = high_output
            # outputs_LSTM = (low_output + high_output) / 2

            # if difficulty == 'Easy':  #, 'Medium', 'Hard'):
            #
            #     all_test_known_logits1 += logits1.cpu().detach().tolist()
            #     all_test_known_logits2 += logits2.cpu().detach().tolist()
            #     all_test_known_logits3 += logits3.cpu().detach().tolist()
            #     all_test_known_logits4 += logits4.cpu().detach().tolist()
            #     all_test_known_logits5 += logits5.cpu().detach().tolist()

        batch_num += 1

        predictions = outputs_ori.data.max(1)[1]
        total += labels.size(0)
        correct += (predictions == labels.data).sum()
        predictions2 = outputs_LSTM.data.max(1)[1]
        total2 += labels.size(0)
        correct2 += (predictions2 == labels.data).sum()

        _pred_k_acc.append(outputs_ori.data.cpu().numpy())
        _pred_k_acc2.append(outputs_LSTM.data.cpu().numpy())
        _pred_k.append(outputs_ori.data.cpu().numpy())
        _pred_k2.append(outputs_LSTM.data.cpu().numpy())
        _labels.append(labels.data.cpu().numpy())

    # if difficulty == 'Hard':
    #     all_data_test_known = np.asarray(all_data_test_known)

    # if difficulty == 'Easy':
    #     all_test_known_logits1 = np.asarray(all_test_known_logits1)
    #     all_test_known_logits2 = np.asarray(all_test_known_logits2)
    #     all_test_known_logits3 = np.asarray(all_test_known_logits3)
    #     all_test_known_logits4 = np.asarray(all_test_known_logits4)
    #     all_test_known_logits5 = np.asarray(all_test_known_logits5)
    #
    #     sio.savemat('all_test_known_logits1.mat', {'all_test_known_logits1': all_test_known_logits1})
    #     sio.savemat('all_test_known_logits2.mat', {'all_test_known_logits2': all_test_known_logits2})
    #     sio.savemat('all_test_known_logits3.mat', {'all_test_known_logits3': all_test_known_logits3})
    #     sio.savemat('all_test_known_logits4.mat', {'all_test_known_logits4': all_test_known_logits4})
    #     sio.savemat('all_test_known_logits5.mat', {'all_test_known_logits5': all_test_known_logits5})

    # if difficulty == 'Hard':
    #     all_data_test_unknown = []

    # if difficulty == 'Easy':
    #     all_test_unknown_logits1_easy = []
    #     all_test_unknown_logits2_easy = []
    #     all_test_unknown_logits3_easy = []
    #     all_test_unknown_logits4_easy = []
    #     all_test_unknown_logits5_easy = []
    # elif difficulty == 'Medium':
    #     all_test_unknown_logits1_medium = []
    #     all_test_unknown_logits2_medium = []
    #     all_test_unknown_logits3_medium = []
    #     all_test_unknown_logits4_medium = []
    #     all_test_unknown_logits5_medium = []
    # elif difficulty == 'Hard':
    #     all_test_unknown_logits1_hard = []
    #     all_test_unknown_logits2_hard = []
    #     all_test_unknown_logits3_hard = []
    #     all_test_unknown_logits4_hard = []
    #     all_test_unknown_logits5_hard = []

    for batch_idx, (data, labels, idx) in enumerate(tqdm(dataloaders['test_unknown'])):
        data, labels = data.cuda(), labels.cuda()

        # print(labels)
        # if difficulty == 'Hard':
        #     all_data_test_unknown += data.detach().cpu().tolist()

        with torch.no_grad():
            # outputs = net(data)
            # outputs_ori, outputs_LSTM, attention_map = net(data)
            # outputs_ori = net(data)
            # outputs_ori, outputs_LSTM = net(data)
            outputs_ori = net(data)
            # outputs_ori, _, _, _, _ = net(data, None)
            # _, _, _, outputs_ori = net(data)
            # outputs_ori, _ = net(data)
            outputs_LSTM = outputs_ori

            # ori_output, low_output, high_output = net(data)
            # outputs_ori = high_output
            # outputs_LSTM = (low_output + high_output) / 2

            # if difficulty == 'Easy':
            #     all_test_unknown_logits1_easy += logits1.cpu().detach().tolist()
            #     all_test_unknown_logits2_easy += logits2.cpu().detach().tolist()
            #     all_test_unknown_logits3_easy += logits3.cpu().detach().tolist()
            #     all_test_unknown_logits4_easy += logits4.cpu().detach().tolist()
            #     all_test_unknown_logits5_easy += logits5.cpu().detach().tolist()
            # elif difficulty == 'Medium':
            #     all_test_unknown_logits1_medium += logits1.cpu().detach().tolist()
            #     all_test_unknown_logits2_medium += logits2.cpu().detach().tolist()
            #     all_test_unknown_logits3_medium += logits3.cpu().detach().tolist()
            #     all_test_unknown_logits4_medium += logits4.cpu().detach().tolist()
            #     all_test_unknown_logits5_medium += logits5.cpu().detach().tolist()
            # elif difficulty == 'Hard':
            #     all_test_unknown_logits1_hard += logits1.cpu().detach().tolist()
            #     all_test_unknown_logits2_hard += logits2.cpu().detach().tolist()
            #     all_test_unknown_logits3_hard += logits3.cpu().detach().tolist()
            #     all_test_unknown_logits4_hard += logits4.cpu().detach().tolist()
            #     all_test_unknown_logits5_hard += logits5.cpu().detach().tolist()

        _pred_u.append(outputs_ori.data.cpu().numpy())
        _pred_u2.append(outputs_LSTM.data.cpu().numpy())

    # if difficulty == 'Hard':
    #     all_data_test_unknown = np.asarray(all_data_test_unknown)

    # if difficulty == 'Easy':
    #     all_test_unknown_logits1_easy = np.asarray(all_test_unknown_logits1_easy)
    #     all_test_unknown_logits2_easy = np.asarray(all_test_unknown_logits2_easy)
    #     all_test_unknown_logits3_easy = np.asarray(all_test_unknown_logits3_easy)
    #     all_test_unknown_logits4_easy = np.asarray(all_test_unknown_logits4_easy)
    #     all_test_unknown_logits5_easy = np.asarray(all_test_unknown_logits5_easy)
    # elif difficulty == 'Medium':
    #     all_test_unknown_logits1_medium = np.asarray(all_test_unknown_logits1_medium)
    #     all_test_unknown_logits2_medium = np.asarray(all_test_unknown_logits2_medium)
    #     all_test_unknown_logits3_medium = np.asarray(all_test_unknown_logits3_medium)
    #     all_test_unknown_logits4_medium = np.asarray(all_test_unknown_logits4_medium)
    #     all_test_unknown_logits5_medium = np.asarray(all_test_unknown_logits5_medium)
    # elif difficulty == 'Hard':
    #     all_test_unknown_logits1_hard = np.asarray(all_test_unknown_logits1_hard)
    #     all_test_unknown_logits2_hard = np.asarray(all_test_unknown_logits2_hard)
    #     all_test_unknown_logits3_hard = np.asarray(all_test_unknown_logits3_hard)
    #     all_test_unknown_logits4_hard = np.asarray(all_test_unknown_logits4_hard)
    #     all_test_unknown_logits5_hard = np.asarray(all_test_unknown_logits5_hard)

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('net1 Acc: {:.5f}'.format(acc))
    acc2 = float(correct2) * 100. / float(total2)
    print('net2 Acc: {:.5f}'.format(acc2))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # # Average precision
    # ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u), list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    # results['AUPR'] = ap_score * 100

    print("net1 Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

    _pred_k2 = np.concatenate(_pred_k2, 0)
    _pred_u2 = np.concatenate(_pred_u2, 0)

    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k2, axis=1), np.max(_pred_u2, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k2, _pred_u2, _labels)

    # # Average precision
    # ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u), list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

    results['ACC'] = acc2
    results['OSCR'] = _oscr_socre * 100.
    # results['AUPR'] = ap_score * 100

    print("net2 Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

    # if difficulty == 'Hard':
    #     np.save('all_data_test_known_hard_aircraft.npy', all_data_test_known)
    #     np.save('all_data_test_unknown_hard_aircraft.npy', all_data_test_unknown)
    # np.save('all_data_test_known_mytry5_20220512_swinB_v3_1.npy', all_data_test_known)
    # np.save('all_data_test_unknown_mytry5_20220512_swinB_v3_1.npy', all_data_test_unknown)

    # if difficulty == 'Easy':
    #     sio.savemat('all_test_unknown_logits1_easy.mat', {'all_test_unknown_logits1_easy': all_test_unknown_logits1_easy})
    #     sio.savemat('all_test_unknown_logits2_easy.mat', {'all_test_unknown_logits2_easy': all_test_unknown_logits2_easy})
    #     sio.savemat('all_test_unknown_logits3_easy.mat', {'all_test_unknown_logits3_easy': all_test_unknown_logits3_easy})
    #     sio.savemat('all_test_unknown_logits4_easy.mat', {'all_test_unknown_logits4_easy': all_test_unknown_logits4_easy})
    #     sio.savemat('all_test_unknown_logits5_easy.mat', {'all_test_unknown_logits5_easy': all_test_unknown_logits5_easy})
    # elif difficulty == 'Medium':
    #     sio.savemat('all_test_unknown_logits1_medium.mat', {'all_test_unknown_logits1_medium': all_test_unknown_logits1_medium})
    #     sio.savemat('all_test_unknown_logits2_medium.mat', {'all_test_unknown_logits2_medium': all_test_unknown_logits2_medium})
    #     sio.savemat('all_test_unknown_logits3_medium.mat', {'all_test_unknown_logits3_medium': all_test_unknown_logits3_medium})
    #     sio.savemat('all_test_unknown_logits4_medium.mat', {'all_test_unknown_logits4_medium': all_test_unknown_logits4_medium})
    #     sio.savemat('all_test_unknown_logits5_medium.mat', {'all_test_unknown_logits5_medium': all_test_unknown_logits5_medium})
    # elif difficulty == 'Hard':
    #     sio.savemat('all_test_unknown_logits1_hard.mat', {'all_test_unknown_logits1_hard': all_test_unknown_logits1_hard})
    #     sio.savemat('all_test_unknown_logits2_hard.mat', {'all_test_unknown_logits2_hard': all_test_unknown_logits2_hard})
    #     sio.savemat('all_test_unknown_logits3_hard.mat', {'all_test_unknown_logits3_hard': all_test_unknown_logits3_hard})
    #     sio.savemat('all_test_unknown_logits4_hard.mat', {'all_test_unknown_logits4_hard': all_test_unknown_logits4_hard})
    #     sio.savemat('all_test_unknown_logits5_hard.mat', {'all_test_unknown_logits5_hard': all_test_unknown_logits5_hard})

