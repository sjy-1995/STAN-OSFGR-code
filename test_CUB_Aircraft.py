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

from data.open_set_datasets import get_class_splits, get_datasets

from config import exp_root

import timm
# from methods.ARPL.arpl_utils import AverageMeter
# from tqdm import tqdm
import numpy as np
# from methods.ARPL.core import evaluation
import evaluation
import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score

import pickle
# from test.utils import closed_set_acc, acc_at_95_tpr, compute_auroc, compute_aupr, compute_oscr
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score

import scipy.io as sio
import math

# swin transformer as the backbone
from swin_transformer import SwinTransformer   # the more complex file

# from openSetClassifier_MoEP_AE_for_fine_grained import openSetClassifier


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
# parser.add_argument('--split_train_val', default=False, type=str2bool,
#                         help='Subsample training set to create validation set', metavar='BOOL')
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
# parser.add_argument('--train_feat_extractor', default=True, type=str2bool,
#                         help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
# parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
#                         help='Do we use softmax or logits for evaluation', metavar='BOOL')

parser.add_argument('--osr_mode', default='max_softmax', type=str, help='{entropy, max_softmax}')

# parser.add_argument('--exp_id', type=str, default='(17.02.2022_|_23.656)')   # cub exp1 448 * 448


# ###################################### self-defined model ######################################

class STAN_OSFGR(nn.Module):  # for swin-B
    def __init__(self, transformer, num_classes=1000):
        super(STAN_OSFGR, self).__init__()
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


for difficulty in ('Easy', 'Medium', 'Hard'):

    # ------------------------
    # DATASETS
    # ------------------------

    args.train_classes, args.open_set_classes = train_classes, open_set_classes[difficulty]

    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            image_size=args.image_size, balance_open_set_eval=False,
                            split_train_val=False, open_set_classes=args.open_set_classes)

    # ------------------------
    # DATALOADERS
    # ------------------------
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = False
        # shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                    shuffle=shuffle, sampler=None, num_workers=args.num_workers)

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

    net = STAN_OSFGR(F, num_classes=len(args.train_classes))

    if args.dataset == 'cub':
        net.load_state_dict(torch.load('CUB_STAN-OSFGR.pth'))  # cub
    elif args.dataset == 'aircraft':
        net.load_state_dict(torch.load('Aircraft_STAN-OSFGR.pth'))   # aircraft

    net = net.to(device)
    net.eval()

    correct, total = 0, 0
    correct2, total2 = 0, 0
    CEloss = nn.CrossEntropyLoss()

    torch.cuda.empty_cache()

    _pred_k_acc, _pred_k, _pred_u, _labels = [], [], [], []
    _pred_k_acc2, _pred_k2, _pred_u2 = [], [], []

    batch_num = 0
    # for data, labels, idx in tqdm(dataloaders['test_known']):
    for batch_idx, (data, labels, idx) in enumerate(dataloaders['test_known']):
        data, labels = data.cuda(), labels.cuda()

        # if difficulty == 'Hard':
        #     all_data_test_known += data.detach().cpu().tolist()

        with torch.no_grad():
            # outputs = net(data)
            # outputs_ori, outputs_LSTM, attention_map = net(data)
            # outputs_ori = net(data)
            # ori_output, low_output, high_output = net(data)
            # outputs_ori = net(data)
            outputs_ori, outputs_LSTM = net(data)
            # outputs_ori = net(data)
            # outputs_ori, _, _, _, _ = net(data, None)
            # _, _, _, outputs_ori = net(data)
            # outputs_ori, _ = net(data)
            # outputs_LSTM = outputs_ori
            # outputs_ori = high_output
            # outputs_LSTM = (low_output + high_output) / 2

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


    # for batch_idx, (data, labels, idx) in enumerate(tqdm(dataloaders['test_unknown'])):
    for batch_idx, (data, labels, idx) in enumerate(dataloaders['test_unknown']):
        data, labels = data.cuda(), labels.cuda()

        with torch.no_grad():
            # outputs = net(data)
            # outputs_ori, outputs_LSTM, attention_map = net(data)
            # outputs_ori = net(data)
            outputs_ori, outputs_LSTM = net(data)
            # outputs_ori = net(data)
            # outputs_ori, _, _, _, _ = net(data, None)
            # _, _, _, outputs_ori = net(data)
            # outputs_ori, _ = net(data)
            # outputs_LSTM = outputs_ori

            # ori_output, low_output, high_output = net(data)
            # outputs_ori = high_output
            # outputs_LSTM = (low_output + high_output) / 2

        _pred_u.append(outputs_ori.data.cpu().numpy())
        _pred_u2.append(outputs_LSTM.data.cpu().numpy())

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

