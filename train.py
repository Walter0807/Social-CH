import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from lib.model.models import Transformer, Discriminator
from lib.utils.utils import *
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import os
import yaml
from pprint import pprint
from easydict import EasyDict
from tqdm import tqdm
from lib.utils.dataset import MPMotion

import ipdb
import argparse
import wandb
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument('--ckpt')
    return parser.parse_args()

def evaluate(model, test_dataloader, epoch):
    print('Eval...')
    eval_frame = [5,10,15,20,25]
    with torch.no_grad():
        model.eval()
        loss_list1={}
        aligned_loss_list1 = {}
        root_loss_list1={}
        for data in tqdm(test_dataloader):
            input_seq, output_seq = data
            input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device) 
            output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) 
            n_joints=int(input_seq.shape[-1]/3)
            use=[input_seq.shape[1]]
            
            input_=input_seq.view(-1,25,input_seq.shape[-1])
            output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])
            input_ = dct.dct(input_)

            rec_ = model.forward(input_[:,1:25,:]-input_[:,:24,:],dct.idct(input_[:,-1:,:]),input_seq,use)
            rec = dct.idct(rec_[-1])
            results = output_[:,:1,:]
            for i in range(1,26):
                results = torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
            results = results[:,1:,:]
            
            prediction_1=results[:,:25,:].view(results.shape[0],-1,n_joints,3) 
            gt_1=output_seq[0][:,1:26,:].view(results.shape[0],-1,n_joints,3)

            for j in eval_frame:
                mpjpe=torch.sqrt(((gt_1[:, :j, :, :] - prediction_1[:, :j, :, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().data.numpy().tolist()
                aligned_loss=torch.sqrt(((gt_1[:, :j, :, :] - gt_1[:, :j, 0:1, :] - prediction_1[:, :j, :, :] + prediction_1[:, :j, 0:1, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().data.numpy().tolist()
                root_loss=torch.sqrt(((prediction_1[:, :j, 0, :] - gt_1[:, :j, 0, :]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().data.numpy().tolist()
                if j not in loss_list1.keys():
                    loss_list1[j] = []
                    aligned_loss_list1[j] = []
                    root_loss_list1[j] = []
                
                loss_list1[j].append(np.mean(mpjpe))
                aligned_loss_list1[j].append(np.mean(aligned_loss))
                root_loss_list1[j].append(np.mean(root_loss))
            
            rec=results[:,:,:]
            rec=rec.reshape(results.shape[0],-1,n_joints,3)
            
            input_seq=input_seq.view(results.shape[0],25,n_joints,3)
            pred=torch.cat([input_seq,rec],dim=1)
            output_seq=output_seq.view(results.shape[0],-1,n_joints,3)[:,1:,:,:]
            gt = torch.cat([input_seq,output_seq],dim=1)

            pred = pred[:,:,:,:].cpu().numpy()
            gt = gt[:,:,:,:].cpu().numpy()

        stats = {}
        for j in eval_frame:
            e1, e2, e3 = np.mean(loss_list1[j])*1000, np.mean(aligned_loss_list1[j])*1000, np.mean(root_loss_list1[j])*1000
            prefix = 'val/frame%d/' % j
            stats[prefix + 'err'] = e1
            stats[prefix + 'err aligned'] = e2
            stats[prefix + 'err root'] = e3
        if epoch >= 0:
            stats['epoch'] = epoch
            wandb.log(stats)
        else:
            pprint(stats)
        return e1, e2, e3
    
opts = parse_args()
with open(opts.config) as f:
    args = yaml.load(f, Loader=yaml.Loader)

pprint(args)
args = EasyDict(args)
set_random_seed(args.seed)

train_data = '%s/training.npy' % opts.data
test_data = '%s/testing.npy' % opts.data
dataset = MPMotion(train_data, concat_last=True)
test_dataset = MPMotion(test_data, concat_last=True)

device = 'cuda'

model = Transformer(d_word_vec=args.d_model, d_model=args.d_model, d_inner=args.d_inner_g,
            n_layers=3, n_head=8, d_k=64, d_v=64, k_levels=args.k_levels, share_d=args.share_d, dropout=args.dropout, device=device).to(device)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

if opts.eval:
    print('Loading checkpoint', opts.ckpt)
    checkpoint = torch.load(opts.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint, strict=True)
    e1, e2, e3 = evaluate(model, test_dataloader, epoch=-1)
    print(e1, e2, e3)

if opts.train:
    wandb.init(
            project="Social-CH",
            config=args,
            name=opts.config
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    real_ = MPMotion(train_data, in_len=0)
    real_motion_dataloader = torch.utils.data.DataLoader(real_, batch_size=args.batch_size, shuffle=True)
    real_motion_all = list(enumerate(real_motion_dataloader))

    discriminator = Discriminator(d_word_vec=45, d_model=45, d_inner=args.d_inner_d, d_hidden=args.d_hidden,
                n_layers=3, n_head=8, d_k=32, d_v=32, dropout=args.dropout, device=device).to(device)

    params = [
        {"params": model.parameters(), "lr": args.lr}
    ]
    optimizer = optim.Adam(params)
    params_d = [
        {"params": discriminator.parameters(), "lr": args.lr}
    ]
    optimizer_d = optim.Adam(params_d)

    min_error = 100000

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epochs):    
        print('Training epoch', epoch)
        model.train()
        discriminator.train()

        if model.training:
            print("Model is in training mode")
        else:
            print("Model is in evaluation mode")

        losses_all = []
        losses_dis = []
        losses_gail = []
        losses_recon = []
        losses_sum = AverageMeter()
        for k in range(args.k_levels + 1):
            losses_all.append(AverageMeter())
            losses_dis.append(AverageMeter())
            losses_gail.append(AverageMeter())
            losses_recon.append(AverageMeter())

        for j, data in tqdm(enumerate(dataloader)):
            
            use = None
            input_seq, output_seq = data 
            B = input_seq.shape[0]

            input_seq = torch.tensor(input_seq,dtype=torch.float32).to(device) 
            output_seq = torch.tensor(output_seq,dtype=torch.float32).to(device) 
            input_ = input_seq.view(-1,25,input_seq.shape[-1]) 
            output_ = output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1, input_seq.shape[-1])
            
            input_dct = dct.dct(input_)
            rec_ = model.forward(input_dct[:,1:25,:]-input_dct[:,:24,:], dct.idct(input_dct[:,-1:,:]), input_seq, use) 

            loss_sum = torch.tensor(0).to(device)
            loss_dis_sum = torch.tensor(0).to(device)

            # Train Discriminator
            for k in range(1, args.k_levels + 1):
                bc = (k==args.k_levels)
                gail = True
                if gail:
                    rec = dct.idct(rec_[k])                          
                    results = output_[:,:1,:]
                    for i in range(1, 1+25):
                        results = torch.cat([results, output_[:,:1,:] + torch.sum(rec[:,:i,:],dim=1, keepdim=True)], dim=1)
                    results = results[:,1:,:]                                                                                               
                    
                    real_full = torch.cat([input_, output_[:,1:,:]], dim=1)
                    pred_full = torch.cat([input_, results.detach()], dim=1)                 

                    if args.gail_sample:
                        sel = random.randint(0, len(real_motion_all) - 1)
                        real_motion = real_motion_all[sel][1][1].float().to(device)
                        B,M,T,D = real_motion.shape
                        real_motion = real_motion.reshape([B*M,T,D])
                    else:
                        real_motion = real_full
                        
                    for param in discriminator.parameters():
                        param.requires_grad = True
                    loss_dis = discriminator.calc_dis_loss(pred_full, real_motion)
                    loss_dis_sum = loss_dis_sum + loss_dis
                    losses_dis[k].update(loss_dis.item(), B)

            optimizer_d.zero_grad()
            loss_dis_sum.backward()
            optimizer_d.step()
            

            for k in range(1, args.k_levels + 1):
                bc = (k==args.k_levels)
                gail = True
                rec = dct.idct(rec_[k])                          # [160, 25, 45]
                if bc:
                    loss_l2 = torch.mean((rec[:,:,:]-(output_[:,1:26,:]-output_[:,:25,:]))**2) 
                    loss_recon = loss_l2
                    losses_recon[k].update(loss_recon.item(), B)
                else:
                    loss_recon = torch.tensor(0).to(device)

                if gail:
                    results = output_[:,:1,:]
                    for i in range(1, 1+25):
                        results = torch.cat([results, output_[:,:1,:] + torch.sum(rec[:,:i,:],dim=1, keepdim=True)], dim=1)
                    results = results[:,1:,:]                                                                                                
                    real_full = torch.cat([input_, output_[:,1:,:]], dim=1)
                    pred_full_grad = torch.cat([input_, results], dim=1)                     
                    if args.gail_sample:
                        sel = random.randint(0, len(real_motion_all) - 1)
                        real_motion = real_motion_all[sel][1][1].float().to(device)
                        B,M,T,D = real_motion.shape
                        real_motion = real_motion.reshape([B*M,T,D])
                    else:
                        real_motion = real_full

                    for p in discriminator.parameters():
                        p.requires_grad = False
                    loss_gail = discriminator.calc_gen_loss(pred_full_grad)
                    losses_gail[k].update(loss_gail.item(), B)   
                else:
                    loss_gail = torch.tensor(0).to(device)
                
                loss_all = args.lambda_recon * loss_recon + args.lambda_gail * loss_gail
                losses_all[k].update(loss_all.item(), B)
                loss_sum = loss_sum + loss_all

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            losses_sum.update(loss_sum.item(), B)

        stats = {}
        for k in range(args.k_levels + 1):
            prefix = 'train/level%d/' % k
            stats[prefix + 'loss_dis'] = losses_dis[k].avg
            stats[prefix + 'loss_gail'] = losses_gail[k].avg
            stats[prefix + 'loss_recon'] = losses_recon[k].avg
            stats[prefix + 'loss_all'] = losses_all[k].avg
        stats['train/loss_sum'] = losses_sum.avg
        stats['epoch'] = epoch
        wandb.log(stats)

        e1, e2, e3 = evaluate(model, test_dataloader, epoch)
        print(e1, e2, e3)
        save_base = 'checkpoint/' + args.expname
        if not os.path.exists(save_base):
            os.mkdir(save_base)
        if (epoch+1) % args.save_freq == 0:
            save_path = save_base + f'/{epoch}.pth'
            torch.save(model.state_dict(), save_path)
        if e1 < min_error:
            min_error = e1
            save_path = save_base + '/best.pth'
            torch.save(model.state_dict(), save_path)
        
        if args.lr_decay is not None:
            # Decay learning rate exponentially
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay
            for param_group in optimizer_d.param_groups:
                param_group['lr'] *= args.lr_decay


            
            
