import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from VBPLayer import VBPLinear, VBPConv



# A net with 2 fully connected VBP layer for regression.
class VBPNetRegression(nn.Module):
    def __init__(self, n_features, beta=1.0, n_hidden=50, C=100):
        super(VBPNetRegression, self).__init__()

        self.beta = beta
        self.C = C

        self.dense1 = VBPLinear(n_features, n_hidden)
        self.dense2 = VBPLinear(n_hidden, 1, isoutput=True)


    def forward(self, inp):
        C = self.C 

        h = self.dense1(inp)
        zh = th.sigmoid(C*h) * h 
        mu = self.dense2(zh)

        v = self.dense1.var(inp, None, C)
        var = self.dense2.var(zh, v, C)

        return mu, var


    def loss(self, data, target, n_train=None):
        C = self.C

        h = self.dense1(data)
        zh = th.sigmoid(C*h) * h 
        mu = self.dense2(zh)

        v = self.dense1.var(data, None, C)
        var = self.dense2.var(zh, v, C)

        KL = sum(l.KL() for l in [self.dense1, self.dense2])

        return 0.5 * self.beta * ((mu - target).pow(2)  + var).mean() + KL/n_train


    def compute_beta(self, loader, iscuda, setbeta=False):
        with th.no_grad():
            coll = 0
            for data, target in loader:
                if iscuda:
                    data, target = data.cuda(), target.cuda()
                mu, var = self.forward(data)
                coll += ((target - mu).pow(2) + var).sum().item()
        if setbeta:
            self.beta = len(loader.dataset)/coll



# A LeNet sized net with convolutional and fully connected layer for classification tasks
class VBPLeNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, sample=True):
        super(VBPLeNet, self).__init__()
        self.sample = sample
        self.n_channels = n_channels

        
        if n_channels == 1:
            self.conv1 = VBPConv(1, 20, 5, stride=2)
            self.conv2 = VBPConv(20, 50, 5, stride=2)
            self.dense1 = VBPLinear(4*4*50, 500)
            self.dense2 = VBPLinear(500, n_classes, isoutput=True)

        elif n_channels == 3:
            self.conv1 = VBPConv(3, 192, 5, stride=2)
            self.conv2 = VBPConv(192, 192, 5, stride=2)
            self.dense1 = VBPLinear(5 * 5 * 192, 1000)
            self.dense2 = VBPLinear(1000, n_classes, isoutput=True)

        self.layers = [self.conv1, self.conv2, self.dense1, self.dense2]

    # Assumg a ReLU forward pass
    def forward(self, data):
        C = 100
        h1 = self.conv1(data); zh1 = th.sigmoid(C*h1) * h1
        h2 = self.conv2(zh1); zh2 = th.sigmoid(C*h2) * h2
        if self.n_channels == 1:
            zh2 = zh2.view(-1, 4*4*50)
        else:
            zh2 = zh2.view(-1, 5*5*192)
        h3 = self.dense1(zh2); zh3 = th.sigmoid(C*h3) * h3
        mu = self.dense2(zh3)
        # variance term
        v = self.conv1.var(data, None, C)
        v = self.conv2.var(zh1, v, C)
        if self.n_channels == 1:
            v = v.view(-1, 4*4*50)
        else:
            v = v.view(-1, 5*5*192)
        v = self.dense1.var(zh2, v, C)
        var = self.dense2.var(zh3, v, C)
        return mu, var

    def loss(self, data, target, n_data):
        mu, var = self.forward(data)

        KL = sum(l.KL() for l in self.layers)

        # Sampling based loss with a single sample
        if self.sample:
            logsoft = F.log_softmax(mu + var.sqrt() * th.rand_like(mu), 1) 
            return -th.sum(target * logsoft, 1).mean() + KL/n_data
        else:
            # Delta method
            p = F.softmax(mu, 1)
            lse = th.logsumexp(mu, 1, True)
            snd = (var * p).sum(1, True) - (var * p.pow(2)).sum(1, True)

            logsftmx = mu - lse - 0.5 * snd
            avgnll = - (target * logsftmx).sum(1).mean()
            return avgnll + KL/n_data

    def predict(self, data, map=True):
        mu, var = self.forward(data)
        if map:
            return F.softmax(mu, 1)
        # Sampling based prediction with a single sample
        if self.sample:
            return F.softmax(mu + var.sqrt() * th.randn_like(mu), 1)
        else:
            p = F.softmax(mu, 1)
            snd = 1 + (var * p.pow(2)).sum(1, True) - (var * p).sum(1, True) + 0.5 * var - 0.5 * (var * p).sum(1, True)
            return p * snd

