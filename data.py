#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_point_cloud_by_axisangle(point_cloud, rotation_angle, rotate_axis = [0,0]):
    # rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
    cosv = np.cos(rotation_angle)
    sinv = np.sin(rotation_angle)
    seita, fai = rotate_axis
    a_x = np.sin(seita) * np.cos(fai)
    a_y = np.sin(seita) * np.sin(fai)
    a_z = np.cos(seita)
    rotate_matrix = np.array([[a_x*a_x*(1-cosv)+cosv, a_x*a_y*(1-cosv)+a_z*sinv, a_x*a_z*(1-cosv)-a_y*sinv],
                            [a_x*a_y*(1-cosv)-a_z*sinv, a_y*a_y*(1-cosv)+cosv, a_y*a_z*(1-cosv)+a_x*sinv],
                            [a_x*a_z*(1-cosv)+a_y*sinv, a_y*a_z*(1-cosv)-a_x*sinv, a_z*a_z*(1-cosv)+cosv]])
    return np.dot(point_cloud.reshape((1024,3)), rotate_matrix)

def random_rotate_batchdata(batch_data):
    B, N, C = batch_data.shape
    result_point = np.zeros((B, N ,C),dtype=np.float32)
    for i in range(B):
        angle = np.random.uniform() * 2 * np.pi
        axis_z = np.random.uniform() *  np.pi
        axis_angle = np.random.uniform() * 2 * np.pi
        rotate_axis = [axis_z,axis_angle]
        result_point[i,...] = rotate_point_cloud_by_axisangle(batch_data[i,...], angle, rotate_axis)
    return result_point
def random_rotate_data(batch_data):
    N, C = batch_data.shape
    result_point = np.zeros((N ,C),dtype=np.float32)

    angle = np.random.uniform() * 2 * np.pi
    axis_z = np.random.uniform() *  np.pi
    axis_angle = np.random.uniform() * 2 * np.pi
    rotate_axis = [axis_z,axis_angle]
    result_point = rotate_point_cloud_by_axisangle(batch_data, angle, rotate_axis)
    return result_point

def rotate_point_cloud_randomangle(batch_data, rotate_pram):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    axis_z, axis_angle, rotate_angle = rotate_pram
    for i in range(batch_data.shape[0]):
        angle = rotate_angle[i]
        rotate_axis = [axis_z[i],axis_angle[i]]
        rotated_data[i,...] = rotate_point_cloud_by_axisangle(batch_data[i,...], angle, rotate_axis)
    return rotated_data

def generate_random_axis(axis_num):
    rotate_angle = np.random.uniform(np.pi/3,2*np.pi/3,size=(axis_num))
    axis_angle = np.random.uniform(np.pi/3,2*np.pi/3,size=(axis_num))
    axis_z = np.random.uniform(np.pi/3,np.pi,size=(axis_num))
    rotate_pram = [axis_z, axis_angle, rotate_angle]
    return rotate_pram

def point_cloud_centerized(batch_data):
    batch_data = np.array(batch_data)
    sphere_core = np.mean(batch_data, axis=1, keepdims=True)#/float(batch_data.shape[1])
    centerize_data = batch_data - sphere_core
    new_core = np.mean(centerize_data, axis=1)#/float(batch_data.shape[1])
    if np.sum(new_core)==0:
        print('yes,It has been centerized succeessfully!')
    return centerize_data

def rotate_point_cloud_by_angle(data, rotation_angle):
    """
    Rotate the point cloud along up direction with certain angle.
    :param batch_data: Nx3 array, original batch of point clouds
    :param rotation_angle: range of rotation
    :return:  Nx3 array, rotated batch of point clouds
    """
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle).astype('float32')
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_rotate_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    #h5_name = '/nfs-data/project/skyler/Pointnet_Pointnet2_pytorch-master/data/ModelnetTest_Rotate10.h5'
    # random rotation file path
    h5_name = DATA_DIR + '/ModelnetTest_Rotate10.h5'
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_rotate_minist_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    h5_name = DATA_DIR+'/MinistTest_Rotate10.h5'
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_data_minist(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, '3d-minist0', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

class ModelNet40(Dataset):
    def __init__(self, num_points, augment = False,partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.augment = augment

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.augment:
            angle = np.random.uniform() * 2 * np.pi
            pointcloud = rotate_point_cloud_by_angle(pointcloud, angle)

        return pointcloud.astype('float32'), label

    def __len__(self):
        return self.data.shape[0]

class ModelNet40Test(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_rotate_data(partition)
        # idx = np.arange(self.data.shape[0])
        # np.random.seed(0)
        # np.random.shuffle(idx)
        # self.data = self.data[idx,...]
        # self.label = self.label[idx,...]
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        # if self.partition == 'train':
        #     angle = np.random.uniform() * 2 * np.pi
        #     pointcloud = rotate_point_cloud_by_angle(pointcloud, angle)
        # if self.partition == 'train':
        #     pointcloud = translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)
        #pointcloud = random_rotate_data(pointcloud)
        return pointcloud.astype('float32'), label

    def __len__(self):
        return self.data.shape[0]

class Minist(Dataset):
    def __init__(self, num_points,augment = False, partition='train'):
        self.data, self.label = load_data_minist(partition)
        self.num_points = num_points
        self.partition = partition
        self.augment = augment

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.augment:
            angle = np.random.uniform() * 2 * np.pi
            pointcloud = rotate_point_cloud_by_angle(pointcloud, angle)
        return pointcloud.astype('float32'), label

    def __len__(self):
        return self.data.shape[0]

class MinistTest(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_rotate_minist_data(partition)
        idx = np.arange(self.data.shape[0])
        np.random.seed(0)
        np.random.shuffle(idx)
        self.data = self.data[idx,...]
        self.label = self.label[idx,...]
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud.astype('float32'), label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    np.random.seed(0)
    Data = MinistTest(1024)

    print(np.random.randn(1))
    #for data, label in train:
    #    print(data.shape)
    #    print(label.shape)
