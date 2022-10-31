# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = 'open_set_recognition'        # directory to store experiment output (checkpoints, logs, etc)
save_dir = 'open_set_recognition/methods/baseline/ensemble_entropy_test'    # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = 'open_set_recognition/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = 'open_set_recognition/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
#cifar_10_root = 'datasets/cifar10'                                          # CIFAR10
#cifar_100_root = 'datasets/cifar100'                                        # CIFAR100
# cub_root = '../../Transformer/newdata/CUB'                                                   # CUB
cub_root = './data/CUB'                                                   # CUB
# aircraft_root = '../../Transformer/newdata/aircraft/fgvc-aircraft-2013b'                      # FGVC-Aircraft
aircraft_root = './data/aircraft/fgvc-aircraft-2013b'                      # FGVC-Aircraft
#mnist_root = 'datasets/mnist/'                                              # MNIST
#pku_air_root = 'datasets/pku-air-300/AIR'                                   # PKU-AIRCRAFT-300
#svhn_root = 'datasets/svhn'                                                 # SVHN
#tin_train_root_dir = 'datasets/tinyimagenet/tiny-imagenet-200/train'        # TinyImageNet Train
#tin_val_root_dir = 'datasets/tinyimagenet/tiny-imagenet-200/val/images'     # TinyImageNet Val

# ----------------------
# FGVC OSR SPLITS
# ----------------------
osr_split_dir = 'data/open_set_splits'

# ----------------------
# PRETRAINED RESNET50 MODEL PATHS (For FGVC experiments)
# Weights can be downloaded from https://github.com/nanxuanzhao/Good_transfer
# ----------------------
# imagenet_moco_path = 'pretrained_models/imagenet/moco_v2_800ep_pretrain.pth.tar'
# places_moco_path = 'pretrained_models/places/moco_v2_places.pth'
places_moco_path = 'pretrained_models/moco_v2_imagenet.pth'
places_supervised_path = 'pretrained_models/places/supervised_places.pth'
imagenet_supervised_path = 'pretrained_models/imagenet/supervised_imagenet.pth'
