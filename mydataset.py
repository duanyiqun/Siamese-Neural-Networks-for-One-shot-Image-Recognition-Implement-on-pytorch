import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image
import torchvision.transforms as transforms


class OmniTrain(Dataset):

    def __init__(self, dataPath, transform=transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])):
        super(OmniTrain, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.imglist, self.num_classes = self.build_index(dataPath)

    def img_loader(self, path):
        degrees  = [0, 90, 180, 270]
        return Image.open(path).rotate(random.choice(degrees)).convert('L')

    def build_index(self, dataPath):
        print("Establish image indeces for training...")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                        #datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                    datas[idx].append(filePath)
                idx += 1
        print("finish creating training dataset ... there are totally {} classes".format(idx))
        return datas, idx

    def __len__(self):
        return  500000

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        image1 = None
        image2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = self.img_loader(random.choice(self.imglist[idx1]))
            image2 = self.img_loader(random.choice(self.imglist[idx1]))
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = self.img_loader(random.choice(self.imglist[idx1]))
            image2 = self.img_loader(random.choice(self.imglist[idx2]))

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class OmniTest(Dataset):

    def __init__(self, dataPath, transform=transforms.ToTensor(), times=200, way=20):
        np.random.seed(1)
        super(OmniTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.build_index(dataPath)
    
    def img_loader(self, path):
        return Image.open(path).convert('L')

    def build_index(self, dataPath):
        print("Establish image indeces to testing...")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                        #datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                    datas[idx].append(filePath)
                idx += 1
        print("finish creating test dataset ... there are totally {} classes".format(idx))
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = self.img_loader(random.choice(self.datas[self.c1]))
            img2 = self.img_loader(random.choice(self.datas[self.c1]))
            label = 1.0
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = self.img_loader(random.choice(self.datas[c2]))
            label = 0.0

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))


# test
if __name__=='__main__':
    trainSet = OmniTrain('./images_background',transform=transforms.ToTensor())
    trainLoader = DataLoader(trainSet, batch_size=2, shuffle=False, num_workers=2)
    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        print(batch_id, img1.size(), img2.size(), label.size())
