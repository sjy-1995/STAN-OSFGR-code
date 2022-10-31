# STAN-OSFGR-code
This is the code for our paper: "Spatial-Temporal Attention Network for Open-Set Fine-Grained Image Recognition", which is under preview of a journal. The authors of this paper are Jiayin Sun, Hong Wang, and Qiulei Dong. The code is based on `https://github.com/sgvaze/osr_closed_set_all_you_need` [1]  

The code is configured by PyTorch 1.7.1, torchvision 0.8.2, Python 3.7.7.

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

To train the model on the Stanford-Cars datasets, run:

```Bash
python train_Stanford_Cars.py
```

To test the trained weights on the Stanford-Cars datasets, run:

```Bash
python test_Stanford_Cars.py
```

the network weights of the STAN-OSFGR model are available at google drive:
CUB: https://drive.google.com/file/d/1j1U-0nC0IMH6S_9VcD_CnF24bNVkE1ur/view?usp=sharing
Aircraft: https://drive.google.com/file/d/1OJrUfes5XBHdFyQyzf8AiJJ2NxlQRBe7/view?usp=sharing
Stanford-Cars: https://drive.google.com/file/d/1vQ5BmvUskp8fFVRHbwB2jzg5azmFyVyg/view?usp=sharing


[1] Vaze S, Han K, Vedaldi A, et al. Open-set recognition: A good closed-set classifier is all you need[C]. ICLR, 2022.
