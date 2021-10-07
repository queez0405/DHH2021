from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from network import GeNet
from dataloader import set_loader
import pdb

import matplotlib.pyplot as plt
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--input_dim',type=int, default=311)
    parser.add_argument('--hid_dim',type=int, default=256)
    parser.add_argument('--n_layers',type=int, default=3)
    parser.add_argument('--n_heads',type=int, default=8)
    parser.add_argument('--pf_dim',type=int, default=512)
    parser.add_argument('--dropout',type=float, default=0.1)
    parser.add_argument('--kfold', type=bool, default = False)
    parser.add_argument('--surv_event',type=bool, default=False)

    # optimization
    parser.add_argument('--optimizer',type=str, default='sgd',choices=['sgd','adam'])
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,70',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    opt = parser.parse_args()

    opt.max_length=opt.input_dim
    opt.save_folder = './save/model/'
    opt.result_path = './save/result/'
    opt.tb_folder = './logger/'

    opt.device = torch.device('cpu')
    if torch.cuda.is_available():
        opt.device = torch.device('cuda',opt.gpu)

    return opt

def set_model(opt):
    model = GeNet(input_dim=opt.input_dim,hid_dim=opt.hid_dim, n_layers=opt.n_layers, n_heads=opt.n_heads, pf_dim=opt.pf_dim, dropout=opt.dropout,max_length=opt.max_length, device=opt.device)
    criterion = torch.nn.L1Loss()

    model.to(opt.device)
    criterion.to(opt.device)

    return model, criterion

def main():
    opt = parse_option()
    
    opt.kfold=False
    train_loader = set_loader(opt)
            
    model,_ = set_model(opt)

    # best model
    ckpt = torch.load('./save/model/no_normalize/kfold_6.pth', map_location=opt.device)
    state_dict=ckpt['model']
    model.load_state_dict(state_dict)
    model.to(opt.device)
    
    for input, answer in train_loader:
        
        with torch.no_grad():
            input=input.to(opt.device)
            answer=answer.to(opt.device)
            
            attn_mask = input.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(opt.device)

            _,score_list = model.encoder(input,attn_mask)
            
            a = score_list[0]
            b = score_list[1]
            c = score_list[2]
            
            alpha = torch.mean(torch.sum(a,dim=1),dim=0).cpu().numpy()
            beta = torch.mean(torch.sum(b,dim=1),dim=0).cpu().numpy()
            gamma = torch.mean(torch.sum(c,dim=1),dim=0).cpu().numpy()
            
            alpha = np.delete(alpha,range(310),0)
            alpha = np.delete(alpha,range(10),1)

            beta = np.delete(beta,range(310),0)
            beta = np.delete(beta,range(10),1)

            gamma = np.delete(gamma,range(310),0)
            gamma = np.delete(gamma,range(10),1)

            total = alpha+beta+gamma
            
            index = np.argsort(-1*total)
            print(index+1)          

if __name__ == '__main__':
    main()
