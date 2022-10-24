# coding: utf-8
import os
import argparse
import datetime
import time
import importlib

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn

import timm
from tqdm import tqdm
import numpy as np
# import evaluation
import sklearn
import sklearn.metrics
# import metrics
import os.path as osp

import math
import random

import inspect
from datetime import datetime

from swin_transformer import SwinTransformer   # the more complex file

# import datasets_unified.utils_version3 as dataHelper
# from utils_MoEP_AE_ResNet18_CEloss import progress_bar
# import json

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='stanford_cars', help="Datasets")
# parser.add_argument('--trial', required=True, type=int, help='Trial number, 0-4 provided')
parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=224)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=600)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=0.3, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='timm_resnet50_pretrained')
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=30)
parser.add_argument('--rand_aug_n', type=int, default=2)

# misc
parser.add_argument('--num_workers', default=2, type=int)
# parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+', help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
# parser.add_argument('--checkpt_freq', type=int, default=20)
parser.add_argument('--checkpt_freq', type=int, default=5)
# parser.add_argument('--checkpt_freq', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
# parser.add_argument('--train_feat_extractor', default=True, type=str2bool, help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
# parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool, help='Do we use softmax or logits for evaluation', metavar='BOOL')


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_networks(networks, result_dir, name='', loss='', criterion=None):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        torch.save(weights, filename)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_experiment(args, runner_name=None):

    args.cuda = torch.cuda.is_available()

    if args.device == 'None':
        args.device = torch.device("cuda:0" if args.cuda else "cpu")
    else:
        args.device = torch.device(args.device if args.cuda else "cpu")

    print(args.gpus)

    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Unique identifier for experiment
    now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
          datetime.now().strftime("%S.%f")[:-3] + ')'

    log_dir = os.path.join(root_dir, 'log', now)
    while os.path.exists(log_dir):
        now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'

        log_dir = os.path.join(root_dir, 'log', now)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    os.mkdir(model_root_dir)

    args.model_dir = model_root_dir

    print(f'Experiment saved to: {args.log_dir}')

    # args.writer = SummaryWriter(log_dir=args.log_dir)
    # hparam_dict = {}
    # for k, v in vars(args).items():
    #     if isinstance(v, (int, float, str, bool, torch.Tensor)):
    #         hparam_dict[k] = v
    # args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})

    print(runner_name)
    print(args)
    return args


def get_optimizer(args, params_list):

    if args.optim is None:

        if options['dataset'] == 'tinyimagenet':
            optimizer = torch.optim.Adam(params_list, lr=args.lr)
        else:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            # optimizer = torch.optim.SGD(params_list, lr=args.lr)

    elif args.optim == 'sgd':

        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == 'adam':

        optimizer = torch.optim.Adam(params_list, lr=args.lr)

    else:

        raise NotImplementedError

    return optimizer

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

        # print('F1_feature_maps.shape is: ', F1_feature_maps.shape)

        F1_splits = F1_feature_maps.view(F1_feature_maps.shape[0], 1024, 49)   # (b, 1024, 196)
        F2_splits = F2_feature_maps.view(F2_feature_maps.shape[0], 1024, 49)   # (b, 1024, 196)
        F3_splits = F3_feature_maps.view(F3_feature_maps.shape[0], 1024, 49)   # (b, 1024, 196)
        F4_splits = F4_feature_maps.view(F4_feature_maps.shape[0], 1024, 49)   # (b, 1024, 196)

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
        LSTM_in_input = torch.cat((splits, Hidden.unsqueeze(1).repeat(1, 49, 1)), 2)   # (b, 196, 1024+1024)
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
args = parser.parse_args()
args.exp_root = 'open_set_recognition'
args.epochs = args.max_epoch
img_size = args.image_size
results = dict()

for i in range(1):

    # ------------------------
    # INIT
    # ------------------------
    if args.feat_dim is None:
        args.feat_dim = 128 if args.model == 'classifier32' else 2048

    args.save_name = '{}_{}_{}'.format(args.model, args.seed, args.dataset)
    runner_name = os.path.dirname(__file__).split("/")[-2:]
    args = init_experiment(args, runner_name=runner_name)

    # ------------------------
    # SEED
    # ------------------------
    seed_torch(args.seed)

    # ------------------------
    # my DATASETS
    # ------------------------
    from torchvision import transforms, utils
    from PIL import Image

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        # mean = [0.4509, 0.4377, 0.4375],
        # std = [0.2865, 0.2851, 0.2930]

    )

    train_preprocess = transforms.Compose([
        # transforms.Resize([280, 280]),
        # transforms.RandomCrop([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        normalize
    ])

    test_preprocess = transforms.Compose([
        # transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize
    ])

    def trainimg_loader(path):
        img_pil = Image.open(path)
        img_pil = img_pil.convert(mode='RGB')
        img_pil = img_pil.resize((224, 224))
        img_tensor = train_preprocess(img_pil)
        return img_tensor

    def testimg_loader(path):
        img_pil = Image.open(path)
        img_pil = img_pil.convert(mode='RGB')
        img_pil = img_pil.resize((224, 224))
        img_tensor = test_preprocess(img_pil)
        return img_tensor

    class trainset(Dataset):
        def __init__(self, loader=trainimg_loader):
            self.file_path = './new_FG_datasets/stanford_cars/mat2txt.txt'
            all_train_val_img_path_list = []
            all_train_val_label_list = []
            all_test_known_img_path_list = []
            all_test_unknown_img_path_list = []
            all_test_known_label_list = []
            all_test_unknown_label_list = []
            with open(self.file_path, 'rb') as f:
                for line in f.readlines():
                    each_line = line.decode().rstrip().split(' ')
                    img_path = './new_FG_datasets/stanford_cars/' + each_line[1]
                    label = int(each_line[2]) - 1  # from 1 -> from 0
                    if_test = int(each_line[3])
                    if label < 98:   # known classes
                        if if_test == 0:   # train images
                            all_train_val_img_path_list.append(img_path)
                            all_train_val_label_list.append(label)

                        elif if_test == 1:   # test known-class images
                            all_test_known_img_path_list.append(img_path)
                            all_test_known_label_list.append(label)

                    else:   # unknown classes
                        if if_test == 1:   # test unknown-class images
                            all_test_unknown_img_path_list.append(img_path)
                            all_test_unknown_label_list.append(label)   # should be >= 98

            train_indice_list = []
            val_indice_list = []
            for each_label in range(98):
                list_indice_this_label = np.where(np.asarray(all_train_val_label_list)==each_label)[0].tolist()
                num_this_label = len(list_indice_this_label)
                train_indice = random.sample(list_indice_this_label, int(0.8 * num_this_label))
                val_indice = list(set(list_indice_this_label) - set(train_indice))
                train_indice_list += train_indice
                val_indice_list += val_indice

            all_train_img_path_list = np.array(all_train_val_img_path_list)[train_indice_list].tolist()
            all_val_img_path_list = np.array(all_train_val_img_path_list)[val_indice_list].tolist()
            all_train_label_list = np.array(all_train_val_label_list)[train_indice_list].tolist()
            all_val_label_list = np.array(all_train_val_label_list)[val_indice_list].tolist()

            self.train_img_path = all_train_img_path_list
            self.val_img_path = all_val_img_path_list
            self.train_labels = all_train_label_list
            self.val_labels = all_val_label_list

            self.train_all_img_path = self.train_img_path + self.val_img_path
            self.train_all_labels = self.train_labels + self.val_labels

            self.test_known_path = all_test_known_img_path_list
            self.test_unknown_path = all_test_unknown_img_path_list
            self.test_known_labels = all_test_known_label_list
            self.test_unknown_labels = all_test_unknown_label_list

            self.loader = loader

        def __getitem__(self, index):
            # fn = self.train_img_path[index]
            fn = self.train_all_img_path[index]
            img = self.loader(fn)
            # target = self.train_labels[index]
            target = self.train_all_labels[index]
            return img, target

        def __len__(self):
            # return len(self.train_img_path)
            return len(self.train_img_path) + len(self.val_img_path)


    mytrainset = trainset()
    # trainloader_ori = DataLoader(dataset=mytrainset, batch_size=args.batch_size, shuffle=True, num_workers=2)


    # def get_mean_std_value(loader):
    #
    #     data_sum, data_squared_sum, num_batches = 0, 0, 0
    #
    #     for data, _ in loader:
    #         # data: [batch_size,channels,height,width]
    #         # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
    #         data_sum += torch.mean(data, dim=[0, 2, 3])  # [batch_size,channels,height,width]
    #         # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
    #         data_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])  # [batch_size,channels,height,width]
    #         # 统计batch的数量
    #         num_batches += 1
    #     # 计算均值
    #     mean = data_sum / num_batches
    #     # 计算标准差
    #     std = (data_squared_sum / num_batches - mean ** 2) ** 0.5
    #     return mean, std
    #
    #
    # mean_values, std_values = get_mean_std_value(trainloader_ori)
    # print('the mean, std of the training dataset are : ', mean_values, std_values)


    class valset(Dataset):
        def __init__(self, loader=testimg_loader):

            self.val_img_path = mytrainset.val_img_path
            self.val_labels = mytrainset.val_labels
            self.loader = loader

        def __getitem__(self, index):
            fn = self.val_img_path[index]
            img = self.loader(fn)
            target = self.val_labels[index]
            return img, target

        def __len__(self):
            return len(self.val_img_path)


    myvalset = valset()


    class testknownset(Dataset):
        def __init__(self, loader=testimg_loader):

            self.test_known_path = mytrainset.test_known_path
            self.test_known_labels = mytrainset.test_known_labels
            self.loader = loader

        def __getitem__(self, index):
            fn = self.test_known_path[index]
            img = self.loader(fn)
            target = self.test_known_labels[index]
            return img, target

        def __len__(self):
            return len(self.test_known_path)


    mytestknownset = testknownset()


    class testunknownset(Dataset):
        def __init__(self, loader=testimg_loader):

            self.test_unknown_path = mytrainset.test_unknown_path
            self.test_unknown_labels = mytrainset.test_unknown_labels
            self.loader = loader

        def __getitem__(self, index):
            fn = self.test_unknown_path[index]
            img = self.loader(fn)
            target = self.test_unknown_labels[index]
            return img, target

        def __len__(self):
            return len(self.test_unknown_path)


    mytestunknownset = testunknownset()

    trainloader = DataLoader(dataset=mytrainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(dataset=myvalset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    knownloader = DataLoader(dataset=mytestknownset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    unknownloader = DataLoader(dataset=mytestunknownset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    # ------------------------
    # SAVE PARAMS
    # ------------------------
    options = vars(args)
    options.update(
        {
            'item': i,
            # 'known': args.train_classes,
            # 'unknown': args.open_set_classes,
            'img_size': img_size,
            # 'dataloaders': dataloaders,
            # 'num_classes': len(args.train_classes)
        }
    )

    dir_name = '{}_{}'.format(options['model'], options['loss'])
    dir_path = os.path.join('/'.join(args.log_dir.split("/")[:-2]), 'results', dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = options['dataset'] + '.csv'
    print('result path:', os.path.join(dir_path, file_name))

torch.manual_seed(options['seed'])
os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
use_gpu = torch.cuda.is_available()
if options['use_cpu']: use_gpu = False

if use_gpu:
    print("Currently using GPU: {}".format(options['gpu']))
    cudnn.benchmark = False
    torch.cuda.manual_seed_all(options['seed'])
else:
    print("Currently using CPU")

# -----------------------------
# MODEL
# -----------------------------
print("Creating model: {}".format(options['model']))

F = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                    embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                    window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    # window_size=14, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False
                    # use_checkpoint=True
                    )  # the feature dim is 1024

# net = mytry5_20220519_v3(F, num_classes=98)
net = STAN_OSFGR(F, num_classes=98)

net_dict = net.state_dict()
pretrained_dict = torch.load('swin_base_patch4_window7_224_22k.pth')['model']
pretrained_dict = {('swinB.'+k): v for k, v in pretrained_dict.items() if (('swinB.'+k) in net_dict) and ('classifier' not in k) and (k not in ['layers.0.blocks.1.attn_mask',
                                                                                                'layers.1.blocks.1.attn_mask',
                                                                                                'layers.2.blocks.1.attn_mask',
                                                                                                'layers.2.blocks.3.attn_mask',
                                                                                                'layers.2.blocks.5.attn_mask',
                                                                                                'layers.2.blocks.7.attn_mask',
                                                                                                'layers.2.blocks.9.attn_mask',
                                                                                                'layers.2.blocks.11.attn_mask',
                                                                                                'layers.2.blocks.13.attn_mask',
                                                                                                'layers.2.blocks.15.attn_mask',
                                                                                                'layers.2.blocks.17.attn_mask'])}
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)


feat_dim = args.feat_dim

options.update(
    {
        'feat_dim': feat_dim,
        'use_gpu': use_gpu
    }
)

# -----------------------------
# PREPARE EXPERIMENT
# -----------------------------
if use_gpu:
    net = net.cuda()

model_path = os.path.join(args.log_dir, 'arpl_models', options['dataset'])
if not os.path.exists(model_path):
    os.makedirs(model_path)


params_list = [{'params': net.parameters()}]
# optimizer = torch.optim.SGD(params_list, lr=0.1, momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.SGD(params_list, lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.SGD(params_list, lr=0.001, momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.AdamW(params_list, lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# optimizer = torch.optim.SGD(params_list, lr=0.5, momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.SGD(params_list, lr=0.1)

start_time = time.time()


def myauroc(inData, outData, in_low=True):
    inDataMin = np.max(inData, 1)
    outDataMin = np.max(outData, 1)
    allData = np.concatenate((inDataMin, outDataMin))
    labels = np.concatenate((np.ones(len(inDataMin)), np.zeros(len(outDataMin))))
    # allData = np.concatenate((1-inDataMin, 1-outDataMin))
    # labels = np.concatenate((np.ones(len(inDataMin)), np.zeros(len(outDataMin))))  # has the same result
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label = in_low)
    return sklearn.metrics.auc(fpr, tpr)


# -----------------------------
# TRAIN
# -----------------------------
global best_auroc
global best_auroc_epoch
global best_auroc_auroc1
global best_auroc_auroc2
global best_auroc_acc1
global best_auroc_acc2
best_auroc = 0
best_auroc_epoch = 0
best_auroc_auroc1 = 0
best_auroc_auroc2 = 0
best_auroc_acc1 = 0
best_auroc_acc2 = 0


for epoch in range(options['max_epoch']):
    print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

    net.train()
    torch.cuda.empty_cache()

    CEloss = nn.CrossEntropyLoss()
    file_name = options['dataset'] + '.csv'
    MSEloss = nn.MSELoss()

    # train
    for batch_idx, (data, labels) in enumerate(tqdm(trainloader)):

        if data.shape[1] == 3:
            pass
        else:
            data = data.repeat(1, 3, 1, 1)

        data = data.cuda(non_blocking=True)
        # labels = torch.Tensor([mapping[x] for x in labels]).long()
        labels = labels.cuda(non_blocking=True)
        outputs_ori, LSTM_logits = net(data)

        loss1 = CEloss(outputs_ori, labels)
        loss2 = CEloss(LSTM_logits, labels)
        loss = loss1 + loss2

        # 2.1 loss regularization
        accumulation_steps = 1
        loss = loss / accumulation_steps
        # 2.2 back propagation

        loss.backward()

        # 3. update parameters of net
        if ((batch_idx + 1) % accumulation_steps) == 0:

            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)

            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient

        print('loss1: {:.6f}; loss2: {:.6f}'.format(loss1, loss2))

    # test
    if epoch % 1 == 0:
    # if epoch % 10 == 0:
        net.eval()
        correct, total = 0, 0
        correct2, total2 = 0, 0

        torch.cuda.empty_cache()

        xK_net1 = []
        yK_net1 = []
        xU_net1 = []
        xK_net2 = []
        yK_net2 = []
        xU_net2 = []

        for batch_idx, (data, labels) in enumerate(tqdm(knownloader)):

            if data.shape[1] == 3:
                pass
            else:
                data = data.repeat(1, 3, 1, 1)

            # labels = torch.Tensor([mapping[x] for x in labels]).long()
            data, labels = data.cuda(), labels.cuda()

            with torch.no_grad():

                outputs_ori, outputs_LSTM = net(data)

                predictions = outputs_ori.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                predictions2 = outputs_LSTM.data.max(1)[1]
                total2 += labels.size(0)
                correct2 += (predictions2 == labels.data).sum()

                xK_net1 += outputs_ori.detach().cpu().tolist()
                yK_net1 += labels.detach().cpu().tolist()
                xK_net2 += outputs_LSTM.detach().cpu().tolist()
                yK_net2 += labels.detach().cpu().tolist()

        xK_net1 = np.asarray(xK_net1)
        yK_net1 = np.asarray(yK_net1)
        xK_net2 = np.asarray(xK_net2)
        yK_net2 = np.asarray(yK_net2)

        for batch_idx, (data, labels) in enumerate(tqdm(unknownloader)):

            if data.shape[1] == 3:
                pass
            else:
                data = data.repeat(1, 3, 1, 1)

            data, labels = data.cuda(), labels.cuda()

            with torch.no_grad():
                outputs_ori, outputs_LSTM = net(data)

                xU_net1 += outputs_ori.detach().cpu().tolist()
                xU_net2 += outputs_LSTM.detach().cpu().tolist()

        xU_net1 = np.asarray(xU_net1)
        xU_net2 = np.asarray(xU_net2)

        # Accuracy
        acc1 = float(correct) * 100. / float(total)
        print('Acc_net1: {:.5f}'.format(acc1))
        acc2 = float(correct2) * 100. / float(total2)
        print('Acc_net2: {:.5f}'.format(acc2))

        auroc1 = myauroc(xK_net1, xU_net1)
        print("net1 Acc (%): {:.3f}\t AUROC (%): {:.3f}\t".format(acc1, auroc1))

        auroc2 = myauroc(xK_net2, xU_net2)
        print("net2 Acc (%): {:.3f}\t AUROC (%): {:.3f}\t".format(acc2, auroc2))

    # if epoch % options['checkpt_freq'] == 0 or epoch == options['max_epoch'] - 1:
    #     save_networks(net, model_path, file_name.split('.')[0] + '_{}'.format(epoch) + 'net1', options['loss'])

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

