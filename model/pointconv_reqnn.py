import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.pointconv_util_reqnn import PointConvDensitySetAbstraction
from model import Qpool

class PointConvDensityClsSsg(nn.Module):
    def __init__(self, num_classes = 40):
        super(PointConvDensityClsSsg, self).__init__()
        self.emb_dims = 1024
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=2, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 1, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=128, in_channel=256 + 1, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)
        self.sa1.apply(self.weights_init)
        self.sa2.apply(self.weights_init)
        self.sa3.apply(self.weights_init)
        self.conv1 = nn.Sequential(nn.Conv1d(1024, self.emb_dims, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(self.emb_dims),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.fc1 = nn.Linear(self.emb_dims, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def qua_trans(self,x):
        batch_size = x.size(0)
        num_points = x.size(2)
        y = x.view(batch_size*3,-1,num_points).cuda()
        return y

    def qua_merge(self,x):
        inSize = x.size()
        batch_size = inSize[0]//3
        x = x.view(batch_size,3,-1)
        y = torch.sum(x*x,dim=1)
        return y

    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight.data,gain=1.0)
        #elif isinstance(m, nn.Conv1d):
        #    nn.init.xavier_uniform(m.weight.data,gain=2.0)
        #elif isinstance(m, nn.Linear):
        #    nn.init.xavier_uniform(m.weight.data,gain=2.0)

    def forward(self, xyz, pca_xyz):
        B, _, _ = xyz.shape
        qmaxpool = Qpool.our_mpool2d.apply
        l1_xyz, l1_points, p1_xyz = self.sa1(xyz, self.qua_trans(xyz), pca_xyz)
        l2_xyz, l2_points, p2_xyz = self.sa2(l1_xyz, l1_points, p1_xyz)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points, p2_xyz)
        qua_points = self.qua_merge(l3_points)
        x = qua_points.view(B, self.emb_dims)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = PointConvDensityClsSsg(num_classes=40)
    output= model(input)
    print(output.size())
