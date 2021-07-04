#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.Qpool as Qpool

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)//3
    ori_dim = x.size(1)
    num_points = x.size(2)
    #### xyz channel merge to get feature knn ####
    x = x.view(3,batch_size,ori_dim,num_points)
    x = x.permute(1, 0, 2, 3).contiguous().view(batch_size,-1,num_points)

    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    #### seperate the xyz channel into a independent condition ####
    feature = feature.view(batch_size, num_points, k, 3, ori_dim).permute(3,0,1,2,4).contiguous().view(-1,num_points,k,ori_dim)
    #print(feature.shape)
    x = x.view(batch_size, num_points, k, 3, ori_dim).permute(3,0,1,2,4).contiguous().view(-1,num_points,k,ori_dim)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
    return feature

class DGCNN_REQNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_REQNN, self).__init__()
        self.args = args
        self.k = args.k
        self.epsilon = 1e-5

        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False)
        self.conv6 = nn.Sequential(nn.Conv1d(1024, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def qua_trans(self,x):
        batch_size = x.size(0)
        num_points = x.size(2)
        xd = x.permute(1,0,2).contiguous()
        y = xd.view(3*batch_size,1,num_points).cuda()
        return y

    def qua_bn(self, x): #batch normalization for quaternion
        xd = x.detach()
        inSize = xd.size()
        batch_size = inSize[0]//3
        xd2 = xd * xd
        xd2 = xd2.view(3,batch_size,inSize[1],inSize[2],inSize[3])
        mean2 = xd2.permute(1,0,2,3,4).contiguous().view(batch_size,-1).mean(dim=1).cuda()
        coefficient = torch.sqrt(mean2 + self.epsilon).view(1,batch_size,1,1,1)
        coefficient = coefficient.repeat(3,1,1,1,1).view(inSize[0],1,1,1).cuda()
        y = torch.div(x, coefficient)
        return y

    def qua_relu(self, x): #relu activation for quaternion
        xd = x.detach()
        inSize = xd.size()
        batch_size = inSize[0]//3
        threshold = 1
        xd = xd.view(3,batch_size,inSize[1],inSize[2],inSize[3]).permute(1,0,2,3,4).contiguous()
        mod = torch.sqrt(torch.sum(xd*xd, dim = 1))

        threshold_mod = copy.deepcopy(mod)
        threshold_mod[:,:,:,:] = threshold
        after_thre_mod= torch.max(threshold_mod,mod)
        coefficient = torch.div(mod,after_thre_mod)
        coefficient = torch.cat((coefficient,coefficient,coefficient),0)
        y = torch.mul(coefficient, x)
        return y

    def qua_merge(self, x):
        inSize = x.size()
        batch_size = inSize[0]//3
        x = x.view(3,batch_size,inSize[1],inSize[2]).permute(1,0,2,3).contiguous()
        y = torch.sum(x*x,dim=1)
        return y

    def forward(self, x):
        qmaxpool = Qpool.our_mpool1d.apply
        batch_size = x.size(0)
        x = self.qua_trans(x)
        x = get_graph_feature(x, k=self.k)
        x = self.qua_relu(self.qua_bn(self.conv1(x)))
        x1 = qmaxpool(x,(1,20),(1,1))

        x = get_graph_feature(x1, k=self.k)
        x = self.qua_relu(self.qua_bn(self.conv2(x)))
        x2 = qmaxpool(x,(1,20),(1,1))

        x = get_graph_feature(x2, k=self.k)
        x = self.qua_relu(self.qua_bn(self.conv3(x)))
        x3 = qmaxpool(x,(1,20),(1,1))

        x = get_graph_feature(x3, k=self.k)
        x = self.qua_relu(self.qua_bn(self.conv4(x)))
        x4 = qmaxpool(x,(1,20),(1,1))

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x = x.view(batch_size*3,1024,1024,-1)
        x = self.qua_relu(self.qua_bn(x))
        x = x.view(batch_size*3,1024,1024)
        x = self.qua_merge(x)
        x = self.conv6(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
