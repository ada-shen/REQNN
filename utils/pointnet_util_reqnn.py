# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import gradcheck
import copy
import os
from time import time
import numpy as np

class our_mpool1d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size = torch.IntTensor([2]), stride = torch.IntTensor([2])):
        x_detach = x.detach()
        inSize = x_detach.size()
        batch_size = inSize[0]//3

        inSize = torch.IntTensor([inSize[0], inSize[1], inSize[2], inSize[3]])

        x_detach = x_detach.view(batch_size,3,inSize[1],inSize[2],inSize[3])
        mod = torch.sqrt(torch.sum(x_detach*x_detach, dim = 1))

        mod = mod.cuda()
        mod, indices = F.max_pool2d(mod, kernel_size, stride, return_indices = True)
        modSize = mod.size()
        ctx.save_for_backward(x, indices)
        x = x.view(inSize[0], inSize[1], -1)
        indices = indices.view(modSize[0],1, modSize[1], -1).repeat(1,3,1,1).view(modSize[0]*3,modSize[1], -1)
        y = torch.gather(x, 2, indices)
        y = y.view(modSize[0] * 3, modSize[1], modSize[2])
        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.cuda()
        x, indices = ctx.saved_variables
        inSize = x.size()

        grad_inputs = copy.deepcopy(x.detach())
        grad_inputs.zero_()
        grad_inputs = grad_inputs.cuda()
        grad_inputs = grad_inputs.view(inSize[0], inSize[1], -1)
        grad_outputs = grad_outputs.view(inSize[0], inSize[1], -1)
        indices = indices.view(inSize[0] // 3, 1, inSize[1], -1).repeat(1,3,1,1).view(inSize[0], inSize[1], -1)
        grad_inputs = grad_inputs.scatter_(2, indices, grad_outputs)
        grad_inputs = grad_inputs.view(inSize[0],inSize[1],inSize[2],inSize[3])
        grad_inputs = grad_inputs.cuda()
        return grad_inputs, None, None

class QBN_RM(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(QBN_RM, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer('moving_mean', torch.zeros(num_features))

    def batch_norm(self, is_training, x, moving_mean, eps, momentum):
        inSize = x.size()
        if not is_training:
            x = torch.div(x, moving_mean.view(1, inSize[1],1,1)).cuda()
        else:
            xd = x.detach()
            batch_size = inSize[0]//3
            mean = (xd*xd).permute(1,0,2,3).contiguous().view(inSize[1],-1).mean(dim=1).cuda()
            mod = torch.sqrt(mean + self.eps)
            x = torch.div(x, mod.view(1, inSize[1],1,1)).cuda()
            moving_mean = momentum * moving_mean + (1-momentum) * mod
        return x, moving_mean


    def forward(self, input):
        if self.moving_mean.device != input.device:
            self.moving_mean = self.moving_mean.to(input.device)
        y, self.moving_mean = self.batch_norm(self.training, input, self.moving_mean, self.eps, self.momentum)
        return y

class QBN_RM1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(QBN_RM1d, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.momentum = momentum
        self.running_average = None

    def forward(self, input):
        xd = input.detach()
        inSize = xd.size()
        batch_size = inSize[0]//3
        xd2 = xd * xd
        mean2 = xd2.permute(1,0,2).contiguous().view(inSize[1],-1).mean(dim=1).cuda()
        mod_xd2 = torch.sqrt(mean2 + self.eps)
        # if not torch.is_tensor(self.running_average):
        #     self.running_average = mod_xd2
        # else:
        #     self.running_average = (1-self.momentum) * self.running_average + self.momentum * mod_xd2
        # coefficient = self.running_average.view(1,inSize[1],1).cuda()
        coefficient = mod_xd2.view(1,inSize[1],1).cuda()
        y = torch.div(input, coefficient)
        return y

def Qrelu(x): #relu activation for quaternion
    xd = x.detach()
    inSize = xd.size()
    batch_size = inSize[0]//3
    threshold = 1
    xd = xd.view(batch_size,3,inSize[1],inSize[2],inSize[3])
    mod = torch.sqrt(torch.sum(xd*xd, dim = 1))

    threshold_mod = copy.deepcopy(mod)
    threshold_mod[:,:,:,:] = threshold
    after_thre_mod= torch.max(threshold_mod,mod)
    coefficient = torch.div(mod,after_thre_mod)
    coefficient = coefficient.view(batch_size,1,inSize[1],inSize[2],inSize[3]).repeat(1,3,1,1,1).view(batch_size*3,inSize[1],inSize[2],inSize[3])
    y = torch.mul(coefficient, x)
    return y

def Qrelu1d(x): #relu activation for quaternion
    xd = x.detach()
    inSize = xd.size()
    batch_size = inSize[0]//3
    threshold = 1
    xd = xd.view(batch_size,3,inSize[1],inSize[2])
    mod = torch.sqrt(torch.sum(xd*xd, dim = 1))

    threshold_mod = copy.deepcopy(mod)
    threshold_mod[:,:,:] = threshold
    after_thre_mod= torch.max(threshold_mod,mod)
    coefficient = torch.div(mod,after_thre_mod)
    coefficient = coefficient.view(batch_size,1,inSize[1],inSize[2]).repeat(1,3,1,1).view(batch_size*3,inSize[1],inSize[2])
    y = torch.mul(coefficient, x)
    return y

def Qmerge(x):
    inSize = x.size()
    batch_size = inSize[0]//3
    x = x.view(batch_size,3,inSize[1],inSize[2])
    y = torch.sum(x*x,dim=1)
    return y

def Qmerge1(x):
    inSize = x.size()
    batch_size = inSize[0]//3
    x = x.view(batch_size,3,inSize[1],inSize[2],inSize[3])
    y = torch.sum(x*x,dim=1)
    return y

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B*C, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx) # [B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    #local coordinate
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # [B, npoint, nsample, C]
    #add a global coordinate
    grouped_xyz = grouped_xyz.permute(0,3,1,2).contiguous().view(B*C,npoint,nsample,-1)
    grouped_xyz_norm = grouped_xyz_norm.permute(0,3,1,2).contiguous().view(B*C,npoint,nsample,-1) # [B*C, npoint, nsample, 1]

    if points is not None:
        idx = idx.view(B,1,npoint,nsample).repeat(1,3,1,1).view(B*3,npoint,nsample)
        grouped_points = index_points(points, idx) #[B*C, npoint, nsample, D]
        #add global coordinate
        new_points = torch.cat([grouped_xyz_norm, grouped_xyz, grouped_points], dim=-1) # [B*C, npoint, nsample, D+2]
        #new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) #[B*C, npoint, nsample, D+1]
    else:
        #new_points = grouped_xyz_norm
        new_points = torch.cat([grouped_xyz_norm, grouped_xyz], dim=-1)

    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    #grouped_xyz = xyz.view(B, 1, N, C)
    grouped_xyz = xyz.permute(0, 2, 1).contiguous()
    grouped_xyz = grouped_xyz.view(B*C, 1, N , -1)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B*3, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstractionQ(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstractionQ, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False))
            self.bn.append(QBN_RM(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.Qmaxpool = our_mpool1d.apply

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B*C, npoint, nsample, D+1]
        new_points = new_points.permute(0, 3, 1, 2) # [B*C, D+1, npoint,nsample]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.bn[i]
            new_points = Qrelu(bn(conv(new_points)))

        new_points = self.Qmaxpool(new_points, (1,self.nsample),(1,self.nsample))
        #new_points = torch.max(new_points, 3)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint+1):
        if i == 0:
            centroid = torch.mean(xyz,axis=1,keepdims=True)
        else:
            centroids[:, i-1] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstractionMsgQ(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgQ, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.Qmaxpool = our_mpool1d.apply
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 1
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
                bns.append(QBN_RM(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            grouped_xyz = grouped_xyz.permute(0,3,1,2).contiguous().view(B*C,S,K,-1)
            if points is not None:
                group_idx = group_idx.view(B,1,S,K).repeat(1,3,1,1).view(B*3,S,K)
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 1, 2)  # [B, D, S, K]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  Qrelu(bn(conv(grouped_points)))

            new_points = self.Qmaxpool(grouped_points,(1,K),(1,1))
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagationQ(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagationQ, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(QBN_RM1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            weight = weight.view(B,1,N,3).repeat(1,3,1,1).view(B*3,N,3)

            idx = idx.view(B,1,N,3).repeat(1,3,1,1).view(B*3,N,3)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B*3, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  Qrelu1d(bn(conv(new_points)))
        return new_points
