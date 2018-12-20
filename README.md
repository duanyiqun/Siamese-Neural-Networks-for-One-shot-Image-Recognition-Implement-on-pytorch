---
layout:     post
title:      Siamese Network structure for one shot learning
subtitle:   用于少样本学习的孪生网络
date:       2018-12-10
author:     BY Duan Yiqun
header-img: img/post-bg-os-metro.jpg

catalog: true
tags:
    - Machine Learning
    - Algorithm 
    - One shot learning

---

This article is a personal note on previous implementations on siamese learning.  Espcecially I took a detailed look at paper [**Siamese-Networks-for-One-Shot-Learning**](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by Univ of Toronto. 

Recently, I meet a project that requires deep learning network structures to learn to judge ICO scam from very limited data. I would like to form it as a one-shot learning rather than zeron shot learning. Siamese network structure is a kind of unique network structure which could used for varification jobs. 

Siamese network structure are first introduced by Yan Leccun in around 1994 [signature-verification-using-a-siamese-time-delay-neural-network](http://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf).  I think the core idea is to use parallel two subgraphs with same parameters or similar parameters and compare the outputs of the twin subgraphs. The comarison could be made by distance,  cross entrophy or others.  The structure could be listed as follows:

[img](https://pic3.zhimg.com/80/v2-33c010a72aeb83a5108263a23a192112_hd.jpg)

However, the paper [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) from Univ of Toronto introduced the siamese network structure again to public in image recognition tasks. 

This note emphasis on paper [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) 

This model based on the Omniglot dataset. There are totaally 20 class of known image data. The requirement is push the model to learn how to recognize these characters from limited 20 images. The 20 class of images are listed as below:


![figure 1](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/1029bb5c9f9cb7ab486eb0c2a2a1c59104820928/1-Figure1-1.png)

### Model Definition

The standard model is a siamese convolutional neural network with $L$ layers each with $ N_l $units, where $h_{1,l}$ represents the hidden vector in layer l for the first twin, and $h_{2,l}$ denotes the same for the second twin. 

<u> Activate Function</u> : The paper use exclusively rectified linear (ReLU) units in the first L − 2 layers and sigmoidal units in the remaining layers. The equation could be summurized as below: 

$a^{(k)}_{1,m} = max-pool(max(0,W^{(k)}_{l−1,l } **h_{1,(l−1) }+ b_l), 2)$ 

,where  ** denotes the convolutional operation.  After the relu, unit, the paper use a max-pooling with kernel size 2, stride 2. 

The paper uses convolutional strcutures as below:

![Figure 4](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/1029bb5c9f9cb7ab486eb0c2a2a1c59104820928/4-Figure4-1.png)

The twin subnetwork share the same convolutional weights and operation, after that, the feature was flattened to 4096 units through fully connected layers. 

I think the core idea of this paper is replace fully connected hidden layers with convolution layers. It's simple but efficient. 

I do a simple re-implement of siamese network structure. Some functions I referenced to [fangping's implementation](https://github.com/fangpin/siamese-pytorch)

### Requirement:
Pytorch 0.4.xx, Torchvision, numpy, PIL, Python 3.5,3.6 tested

### How to use:
```sh
usage: train.py [-h] [--train_path TRAIN_PATH] [--test_path TEST_PATH]
                [--way WAY] [--times TIMES] [--workers WORKERS]
                [--batch_size BATCH_SIZE] [--lr LR] [--max_iter MAX_ITER]
                [--save_path SAVE_PATH]

PyTorch One shot siamese training

optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        training folder
  --test_path TEST_PATH
                        path of testing folder
  --way WAY             how much way one-shot learning
  --times TIMES         number of samples to test accuracy
  --workers WORKERS     number of dataLoader workers
  --batch_size BATCH_SIZE
                        number of batch size
  --lr LR               learning rate
  --max_iter MAX_ITER   number of iterations before stopping
  --save_path SAVE_PATH
                        path to save model
```

