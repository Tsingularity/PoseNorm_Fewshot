import numpy as np
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torch.nn import NLLLoss,BCEWithLogitsLoss
from . import util

def train_stage_1(train_loader,model,
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
            log_prediction = model.forward_stage_1(image_inp,mask)

        elif isinstance(inp,torch.Tensor):
            inp = inp.cuda()
            log_prediction = model.forward_stage_1(inp)
        
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

    writer.add_scalar('dynamic_loss',avg_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc


def train_PN_stage_1(train_loader,model,
    optimizer,writer,iter_counter,alpha):

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr',lr,iter_counter)
    criterion = NLLLoss().cuda()
    criterion_part = BCEWithLogitsLoss().cuda()

    avg_dynamic_loss = 0
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
            
        log_prediction,heatmap_logits = model.forward_stage_1(inp,mask)
        
        loss_heatmap = criterion_part(heatmap_logits,mask)
        loss_dynamic = criterion(log_prediction,target)
        loss = alpha*loss_heatmap+loss_dynamic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*(torch.sum(torch.eq(max_index,target)).float()/batch_size).item()

        avg_acc += acc
        avg_total_loss += loss.item()
        avg_dynamic_loss += loss_dynamic.item()
        avg_heatmap_loss += loss_heatmap.item()

    avg_total_loss = avg_total_loss/(i+1)
    avg_dynamic_loss = avg_dynamic_loss/(i+1)
    avg_heatmap_loss = avg_heatmap_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('total_loss',avg_total_loss,iter_counter)
    writer.add_scalar('dynamic_loss',avg_dynamic_loss,iter_counter)
    writer.add_scalar('heatmap_loss',avg_heatmap_loss,iter_counter)

    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc


def train_stage_2(train_loader,model,
    optimizer,writer,iter_counter):

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr',lr,iter_counter)
    criterion = NLLLoss().cuda()

    num_fake_novel_class = model.num_fake_novel_class
    shots = model.shots[0]
    num_class = model.num_class
    way = model.way

    avg_loss = 0
    avg_acc = 0

    for i, (inp,target) in enumerate(train_loader):

        iter_counter += 1

        fake_novel_class_id = target.view(way,shots)[:num_fake_novel_class,0]
        fake_novel_class_id_list = fake_novel_class_id.tolist()
        fake_novel_class_id = fake_novel_class_id.cuda()

        fake_base_class_id_list = []
        
        for j in range(num_class):
            if j not in fake_novel_class_id_list:
                fake_base_class_id_list.append(j)
        fake_base_class_id = torch.tensor(fake_base_class_id_list).long().cuda()

        target_for_loss = target.view(way,shots)[:,5:].cuda()
        target_for_loss = target_for_loss.view(-1)

        if isinstance(inp,list):
            (image_inp,mask) = inp
            image_inp = image_inp.cuda()
            mask = mask.cuda()
            feature_vector = model.get_feature_vector(image_inp,mask)

        elif isinstance(inp,torch.Tensor):
            inp = inp.cuda()
            feature_vector = model.get_feature_vector(inp)

        log_prediction = model.forward_stage_2(feature_vector,fake_novel_class_id,fake_base_class_id)
        loss = criterion(log_prediction,target_for_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        _,max_index = torch.max(log_prediction,1)
        acc = 100*(torch.sum(torch.eq(max_index,target_for_loss)).float()/target_for_loss.size(0)).item()

        avg_acc += acc
        avg_loss += loss_value

    avg_loss = avg_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('dynamic_loss',avg_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc