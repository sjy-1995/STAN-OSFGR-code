# STAN-OSFGR-code
This is the code for our paper: "Hierarchical Attention Network for Open-Set Fine-Grained Image Recognition"[1], which has been accepted by T-CSVT. The code is based on `https://github.com/sgvaze/osr_closed_set_all_you_need` [2]  

In addition, the Supplementary Material of the paper has been provided in "Supplementary Material of the Paper: Hierarchical Attention Network for Open-Set Fine-Grained Image Recognition.pdf".

## Environments
The code is configured by PyTorch 1.7.1, torchvision 0.8.2, Python 3.7.7.

## Datasets Preparation  
### Stanford-Cars
Please download the original Stanford-Cars datasets from `https://ai.stanford.edu/~jkrause/cars/car_dataset.html`,  put the dataset in the "./dataset/stanford_cars" folder, and run the two steps in "process.py" in the subfolder "./dataset/stanford_cars/"  for processing the format.

### CUB
Please download the dataset from `http://www.vision.caltech.edu/visipedia/CUB-200.html`

### Aircraft
Please download the dataset from `https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/`

## Trained Weights of the Proposed STAN-OSFGR

### Stanford-Cars
Please download from `https://drive.google.com/file/d/1vQ5BmvUskp8fFVRHbwB2jzg5azmFyVyg/view?usp=sharing`

### CUB
Please download from `https://drive.google.com/file/d/1j1U-0nC0IMH6S_9VcD_CnF24bNVkE1ur/view?usp=sharing`

### Aircraft
Please download from `https://drive.google.com/file/d/1OJrUfes5XBHdFyQyzf8AiJJ2NxlQRBe7/view?usp=sharing`

## Run

### CUB/Aircraft

To train the model on the CUB/Aircraft datasets, run:

```Bash
python train_CUB_Aircraft.py
```

To test the trained weights on the CUB datasets, run:

```Bash
python test_CUB_Aircraft.py --dataset cub
```

To test the trained weights on the Aircraft datasets, run:

```Bash
python test_CUB_Aircraft.py --dataset aircraft
```

### Stanford-Cars

To train the model on the Stanford-Cars datasets, run:

```Bash
python train_Stanford_Cars.py
```

To test the trained weights on the Stanford-Cars datasets, run:

```Bash
python test_Stanford_Cars.py
```

[1] Jiayin Sun, Hong Wang, Qiulei Dong. Hierarchical Attention Network for Open-Set Fine-Grained Image Recognition[J]. IEEE Transactions on Circuits and Systems for Video Technology, DOI: 10.1109/TCSVT.2023.3325001, 2023.

[2] Vaze S, Han K, Vedaldi A, et al. Open-set recognition: A good closed-set classifier is all you need[C]. ICLR, 2022.
