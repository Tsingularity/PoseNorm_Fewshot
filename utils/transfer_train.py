import numpy as np
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torch.nn import NLLLoss,BCEWithLogitsLoss
from . import util

def default_train(train_loader,model,
    optimizer,writer,iter_counter):

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr',lr,iter_counter)
    criterion = NLLLoss().cuda()

    avg_loss = 0
    avg_acc = 0

    for i, (inp,target) in enumerate(train_loader):

        iter_counter += 1
        batch_size = target.size(0)
        target = target.cuda()

        if isinstance(inp,list):
            (image_inp,mask) = inp
            image_inp = image_inp.cuda()
            mask = mask.cuda()
            log_prediction = model(image_inp,mask)

        elif isinstance(inp,torch.Tensor):
            inp = inp.cuda()
            log_prediction = model(inp)
        
        loss = criterion(log_prediction,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*(torch.sum(torch.eq(max_index,target)).float()/batch_size).item()

        avg_acc += acc
        avg_loss += loss.item()

    avg_loss = avg_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('transfer_loss',avg_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc


def PN_train(train_loader,model,
    optimizer,writer,iter_counter,alpha):
    
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr',lr,iter_counter)
    criterion = NLLLoss().cuda()
    criterion_part = BCEWithLogitsLoss().cuda()

    avg_transfer_loss = 0
    avg_heatmap_loss = 0
    avg_total_loss = 0
    avg_acc = 0

    for i, ((inp,mask),target) in enumerate(train_loader):

        iter_counter += 1
        batch_size = target.size(0)

        inp = inp.cuda()
        mask = mask.cuda()

        target = target.cuda()

        if iter_counter%1000==0:
            model.eval()
            util.visualize(model,writer,iter_counter,inp[:9],mask[:9])
            model.train()
            
        log_prediction,heatmap_logits = model(inp,mask)
        
        loss_heatmap = criterion_part(heatmap_logits,mask)
        loss_transfer = criterion(log_prediction,target)
        loss = alpha*loss_heatmap+loss_transfer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*(torch.sum(torch.eq(max_index,target)).float()/batch_size).item()

        avg_acc += acc
        avg_total_loss += loss.item()
        avg_transfer_loss += loss_transfer.item()
        avg_heatmap_loss += loss_heatmap.item()

    avg_total_loss = avg_total_loss/(i+1)
    avg_transfer_loss = avg_transfer_loss/(i+1)
    avg_heatmap_loss = avg_heatmap_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('total_loss',avg_total_loss,iter_counter)
    writer.add_scalar('transfer_loss',avg_transfer_loss,iter_counter)
    writer.add_scalar('heatmap_loss',avg_heatmap_loss,iter_counter)

    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc