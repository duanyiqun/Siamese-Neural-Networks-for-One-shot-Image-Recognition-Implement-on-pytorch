from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from mydataset import OmniTrain, OmniTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Siamese
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from utils import progress_bar
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time


parser = argparse.ArgumentParser(description='PyTorch One shot siamese training ')

parser.add_argument("--train_path",default="./images_background", help="training folder")
parser.add_argument("--test_path", default="./images_evaluation", help='path of testing folder')
parser.add_argument("--way", default=20, type=int, help="how much way one-shot learning")
parser.add_argument("--times", default=400, type=int,help="number of samples to test accuracy")
parser.add_argument("--workers", default=2, type=int,help="number of dataLoader workers")
parser.add_argument("--batch_size", default=128, type=int,help="number of batch size")
parser.add_argument("--lr", default=0.1,type=float, help="learning rate")
parser.add_argument("--max_iter", default=50000,type=int, help="number of iterations before stopping")
parser.add_argument("--save_path", default="./model/siamese", help="path to store model")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..\n')

#trainloader, testloader = dataset.cifar10.process()
trainSet = OmniTrain(args.train_path)
testSet = OmniTest(args.test_path, times = args.times, way = args.way)

trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)
print('==> Index established.\n')

loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
net = Siamese()
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

def train():
    net.train()
    train_loss = []
    loss_val = 0
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if batch_id > args.max_iter:
            break
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.data[0]
        loss.backward()
        optimizer.step()
        if batch_id % 5 == 0 :
            print('batch [%d]\tloss:\t%.5f\t'%(batch_id, loss_val/5, ))
            train_loss.append(loss_val)
            loss_val = 0
        if batch_id % 1000 == 0:
            right, error = 0, 0
            for _, (test1, test2, _) in enumerate(testLoader, 1):
                test1, test2 = test1.to(device), test2.to(device)
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else: error += 1
            print('*'*70)
            print('[%d]\tright:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            acc = right*1.0/(right+error)
            if best_acc < acc:
                best_acc = acc
                state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': batch_id,
                }
                savepath = os.path.join(args.save_path, 'bestcheck.plk')
                torch.save(state, savepath)

            
        

if __name__ == '__main__':
    train()
    print(best_acc)