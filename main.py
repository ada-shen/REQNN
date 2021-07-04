#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import argparse
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import ModelNet40, random_rotate_batchdata, Minist, ModelNet40Test, MinistTest
from model.dgcnn_ori import DGCNN_ORI
from model.dgcnn_reqnn import DGCNN_REQNN
from model.pointnet2 import PointNet2_ORI, PointNet2_REQNN
from model.pointconv_ori import PointConvDensityClsSsg as PointConv_ORI
from model.pointconv_reqnn import PointConvDensityClsSsg as PointConv_REQNN
from util import cal_loss, IOStream, get_pca_xyz

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    if args.dataset =='modelnet':
        num_class = 40
        DATACLASS = ModelNet40
    elif args.dataset == 'mnist':
        num_class = 10
        DATACLASS = Minist
    train_loader = DataLoader(DATACLASS(partition='train',augment=args.augment, num_points=args.num_points), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(DATACLASS(partition='test',augment=args.augment ,num_points=args.num_points), num_workers=8,
                             batch_size=8, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    if args.model == 'dgcnn_ori':
        model = DGCNN_ORI(args, output_channels=num_class).to(device)
    elif args.model == 'dgcnn_reqnn':
        model = DGCNN_REQNN(args, output_channels=num_class).to(device)
    elif args.model == 'pointnet2_ori':
        model = PointNet2_ORI(num_class).to(device)
    elif args.model == 'pointnet2_reqnn':
        model = PointNet2_REQNN(num_class).to(device)
    elif args.model == 'pointconv_ori':
        model = PointConv_ORI(num_class).to(device)
    elif args.model == 'pointconv_reqnn':
        model = PointConv_REQNN(num_class).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr/100)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss

    best_test_acc = 0

    for epoch in range(args.epochs):
        scheduler.step()
        # Train
        model.train()
        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []

        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            if args.model == 'pointconv_reqnn':
                pca_xyz = get_pca_xyz(np.array(data.permute(0, 2, 1).detach().cpu()))
                pca_xyz = torch.from_numpy(pca_xyz).float().cuda()
                logits = model(data, pca_xyz)
            else:
                logits = model(data)
            #logits  = model(data)

            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),                                                                   metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        if epoch % 3 == 0:
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            for data, label in test_loader:
                data = torch.from_numpy(random_rotate_batchdata(data)).float()
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                if args.model == 'pointconv_reqnn':
                    pca_xyz = get_pca_xyz(np.array(data.permute(0, 2, 1).detach().cpu()))
                    pca_xyz = torch.from_numpy(pca_xyz).float().cuda()
                    logits = model(data,pca_xyz)
                else:
                    logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                  test_loss*1.0/count,
                                                                                  test_acc,
                                                                                  avg_per_class_acc)
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/model.t7' % args.exp_name)

        if epoch+1 == args.epochs:
            torch.save(model.state_dict(), 'checkpoints/%s/last_epoch_model.t7' % args.exp_name)

def test(args, io):
    if args.dataset =='modelnet':
        num_class = 40
        DATACLASS = ModelNet40Test
    elif args.dataset == 'mnist':
        num_class = 10
        DATACLASS = MinistTest

    test_loader = DataLoader(DATACLASS(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=8, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    if args.model == 'dgcnn_ori':
        model = DGCNN_ORI(args, output_channels=num_class).to(device)
    elif args.model == 'dgcnn_reqnn':
        model = DGCNN_REQNN(args, output_channels=num_class).to(device)
    elif args.model == 'pointnet2_ori':
        model = PointNet2_ORI(num_class).to(device)
    elif args.model == 'pointnet2_reqnn':
        model = PointNet2_REQNN(num_class).to(device)
    elif args.model == 'pointconv_ori':
        model = PointConv_ORI(num_class).to(device)
    elif args.model == 'pointconv_reqnn':
        model = PointConv_REQNN(num_class).to(device)
    else:
        raise Exception("Not implemented")
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        if args.model == 'pointconv_reqnn':
            pca_xyz = get_pca_xyz(np.array(data.permute(0, 2, 1).detach().cpu()))
            pca_xyz = torch.from_numpy(pca_xyz).float().cuda()
            logits = model(data, pca_xyz)
        else:
            logits = model(data)

        #logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# define bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Here we show the unique arguments (we used in training) which are different when training different models.

# dgcnn reqnn
# --exp_name dgcnn_reqnn_1024_train --model dgcnn_reqnn --use_sgd=True --lr=0.1
# dgcnn ori without rotation augment
# --exp_name dgcnn_ori_nw_1024_train --model dgcnn_ori --use_sgd=True --lr=0.1
# dgcnn ori with rotation augment
# --exp_name dgcnn_ori_w_1024_train --model dgcnn_ori --use_sgd=True --lr=0.1 --augment True

# pointconv reqnn
# -exp_name pointconv_reqnn_1024_train --model pointconv_reqnn --use_sgd=True --lr=0.01
# pointconv without rotation augment
# -exp_name pointconv_ori_nw_1024_train --model pointconv_ori --use_sgd=True --lr=0.01
# pointconv with rotation augment
# -exp_name pointconv_ori_w_1024_train --model pointconv_ori --use_sgd=True --lr=0.01 --augment True

# pointnet++ reqnn
# -exp_name pointnet2_reqnn_1024_train --model pointnet2_reqnn --use_sgd=False --lr=0.0001
# pointnet++ without rotation augment
# -exp_name pointnet2_ori_nw_1024_train --model pointnet2_ori --use_sgd=False --lr=0.0001
# pointnet++ with rotation augment
# -exp_name pointnet2_ori_w_1024_train --model pointnet2_ori --use_sgd=False --lr=0.0001 --augment True

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Classification')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use (default: GPU 0)')
    parser.add_argument('--exp_name', type=str, default='dgcnn_reqnn_1024_train', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn_reqnn', help='Model to use, [pointnet2_reqnn, pointnet2_ori, '
                                                                         'dgcnn_reqnn, dgcnn_ori, pointconv_reqnn, pointconv_ori]')
    parser.add_argument('--dataset', type=str, default='modelnet', help='dataset to use, [modelnet, mnist]')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, help='number of episode to train')
    parser.add_argument('--use_sgd', type=str2bool, default=True, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,  help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=str2bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--eval', type=str2bool, default=False, help='evaluate the model')
    parser.add_argument('--augment', type=str2bool, default=False, help='augment data by random z-axis rotating')
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, help='Num of nearest neighbors to use')
    parser.add_argument('--eval_model_path', type=str, default='', help='your trained model path to eval')
    args = parser.parse_args()

    _init_()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    io = IOStream('checkpoints/' + args.exp_name + '/train.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
