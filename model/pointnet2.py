import numpy as np
import copy

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils.pointnet_util_reqnn import PointNetSetAbstractionQ, Qmerge, Qmerge1, PointNetFeaturePropagationQ, PointNetSetAbstractionMsgQ
from utils.pointnet_util_ori import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation

class PointNet2_REQNN(nn.Module):
    def __init__(self,num_class):
        super(PointNet2_REQNN, self).__init__()
        self.sa1 = PointNetSetAbstractionQ(npoint=512, radius=0.2, nsample=32, in_channel=2, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstractionQ(npoint=128, radius=0.4, nsample=64, in_channel=128+2 , mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstractionQ(npoint=None, radius=None, nsample=128, in_channel=256+1 , mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz,None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = Qmerge(l3_points).view(B,1024)
        #x = self.conv(x).view(B,1024)

        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        return x


class PointNet2_ORI(nn.Module):
    def __init__(self,num_class):
        super(PointNet2_ORI, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 6, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2))
        x = self.fc3(x)
        return x



if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = PointNet2_REQNN(num_classes=40)
    output= model(input)
    print(output.size())
