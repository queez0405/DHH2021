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
from sklearn.metrics import r2_score 


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
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
    parser.add_argument('--kfold', type=bool, default = True)
    parser.add_argument('--surv_event', type=bool, default = False)

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

    # load model
    parser.add_argument('--load_folder',default='', type=str)
    parser.add_argument('--test',type=bool, default=False)

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


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (input, answer) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        input=input.to(opt.device)
        answer=answer.to(opt.device)
        
        attn_mask = input.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask.to(opt.device)

        # compute loss
        output = model(input,attn_mask)
        loss = criterion(output, answer)

        # update metric
        losses.update(loss.item())

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # print info
    print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    sys.stdout.flush()    
            
    return losses.avg

def test(opt,loader, model,criterion, epoch):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    maelosses = AverageMeter()
    mselosses = AverageMeter()
    rmslosses = AverageMeter()
    r2_scores = AverageMeter()

    mse = torch.nn.MSELoss()

    end = time.time()
    for input, answer in loader:
        data_time.update(time.time() - end)
        
        with torch.no_grad():
            input=input.to(opt.device)
            answer=answer.to(opt.device)

            attn_mask = input.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(opt.device)

            output = model(input,attn_mask)
            maeloss = criterion(output, answer)
            mseloss = mse(output,answer)
            r2 = r2_score(answer.cpu(),output.cpu())

        # update metric
        maelosses.update(maeloss.item())
        mselosses.update(mseloss.item())
        rmslosses.update((torch.sqrt(mseloss)+0.0001).item())
        r2_scores.update(r2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # print info
    print('Test: [{0}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'MAEloss {maeloss.val:.3f} ({maeloss.avg:.3f})\t'
                  'MSEloss {mseloss.val:.3f} ({mseloss.avg:.3f})\t'
                  'RMSloss {rmsloss.val:.3f} ({rmsloss.avg:.3f})\t'
                  'R2_Score {r2score.val:.3f} ({r2score.avg:.3f})'.format(
                   epoch, opt.k_index, batch_time=batch_time,
                   data_time=data_time, maeloss=maelosses,mseloss=mselosses,rmsloss=rmslosses,r2score=r2_scores))
    sys.stdout.flush()    
    
    return maelosses.avg, mselosses.avg, rmslosses.avg, r2_scores.avg

def main():
    opt = parse_option()
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    if not opt.test :
        losses = 0
        for k_index in range(1,10):
            opt.k_index = k_index
            train_loader, test_loader = set_loader(opt)
            model, criterion = set_model(opt)
            optimizer = set_optimizer(opt, model)

            loss = 1000000

            for epoch in range(1, opt.epochs + 1):

                # train
                time1 = time.time()
                train_loss= train(train_loader, model, criterion, optimizer, epoch, opt)
                time2 = time.time()
                print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

                test_loss,_,_,_ = test(opt,test_loader,model,criterion,epoch)

                # tensorboard logger
                logger.log_value('train_loss', train_loss, epoch)
                logger.log_value('test_loss', test_loss, epoch)
                logger.log_value('learning_ratte', optimizer.param_groups[0]['lr'], epoch)

                if test_loss < loss :
                    loss = test_loss
                    save_file = os.path.join(
                        opt.save_folder, 'kfold_{fold}.pth'.format(fold=k_index))
                    save_model(model, optimizer, opt, epoch, save_file)
            losses += loss
            print(str(k_index),':', loss)
        print("Average:", losses/10)
    else :
        mse_losses = 0
        mae_losses = 0
        rms_losses = 0
        r2_scores = 0
        for k_index in range(1,10):
            opt.k_index = k_index
            train_loader, test_loader = set_loader(opt)
            model = GeNet(input_dim=opt.input_dim,hid_dim=opt.hid_dim, n_layers=opt.n_layers, n_heads=opt.n_heads, pf_dim=opt.pf_dim, dropout=opt.dropout,max_length=opt.max_length, device=opt.device)
            ckpt = torch.load('./save/model/'+opt.load_folder+'/kfold_'+str(k_index)+'.pth', map_location=opt.device)
            state_dict=ckpt['model']
            model.load_state_dict(state_dict)
            model.to(opt.device)
            criterion = torch.nn.L1Loss()
            optimizer = set_optimizer(opt, model)

            test_mae, test_mse, test_rms, r2 = test(opt,test_loader,model,criterion,0)
        
            mse_losses += test_mse
            mae_losses += test_mae
            rms_losses += test_rms
            r2_scores += r2
            print(str(k_index))
            print("mse:",test_mse)
            print("mae:",test_mae)
            print("rms:",test_rms)
            print("r2_score",r2)
        print("Average")
        print("mse:",mse_losses/10)
        print("mae:",mae_losses/10)
        print("rms:",rms_losses/10)
        print("r2_score:",r2_scores/10)

if __name__ == '__main__':
    main()
