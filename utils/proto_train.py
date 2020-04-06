import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import NLLLoss,BCEWithLogitsLoss,BCELoss
from . import util

def default_train(train_loader,model,
    optimizer,writer,iter_counter):

    way = model.way
    test_shot = model.shots[-1]
    target = torch.LongTensor([i//test_shot for i in range(test_shot*way)]).cuda()
    criterion = NLLLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)

    avg_loss = 0
    avg_acc = 0

    for i, (inp,_) in enumerate(train_loader):

        iter_counter += 1

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

        loss_value = loss.item()
        _,max_index = torch.max(log_prediction,1)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/test_shot/way

        avg_acc += acc
        avg_loss += loss_value

    avg_acc = avg_acc/(i+1)
    avg_loss = avg_loss/(i+1)

    writer.add_scalar('proto_loss',avg_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc


def PN_train(train_loader,model,
    optimizer,writer,iter_counter,alpha):

    test_shot = model.shots[-1]
    way = model.way

    target = torch.LongTensor([i//test_shot for i in range(test_shot*way)]).cuda()
    criterion = NLLLoss().cuda()
    criterion_part = BCEWithLogitsLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)

    avg_proto_loss = 0
    avg_heatmap_loss = 0
    avg_total_loss = 0
    avg_acc = 0

    for i, ((inp,mask),_) in enumerate(train_loader):

        iter_counter += 1      
        inp = inp.cuda()
        mask = mask.cuda()

        if iter_counter%1000==0:
            model.eval()
            util.visualize(model,writer,iter_counter,inp[:9],mask[:9])
            model.train()
        
        log_prediction,heatmap_logits = model(inp,mask)

        loss_heatmap = criterion_part(heatmap_logits,mask)
        loss_proto = criterion(log_prediction,target)
        loss = alpha*loss_heatmap+loss_proto
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/test_shot/way

        avg_acc += acc
        avg_total_loss += loss.item()
        avg_proto_loss += loss_proto.item()
        avg_heatmap_loss += loss_heatmap.item()

    avg_total_loss = avg_total_loss/(i+1)
    avg_proto_loss = avg_proto_loss/(i+1)
    avg_heatmap_loss = avg_heatmap_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('total_loss',avg_total_loss,iter_counter)
    writer.add_scalar('proto_loss',avg_proto_loss,iter_counter)
    writer.add_scalar('heatmap_loss',avg_heatmap_loss,iter_counter)

    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc



def PN_train_less_annot(train_loader,model,
    optimizer,writer,iter_counter,alpha,batch_size):

    test_shot = model.shots[-1]
    way = model.way

    target = torch.LongTensor([i//test_shot for i in range(test_shot*way)]).cuda()
    criterion = NLLLoss().cuda()
    criterion_part = BCEWithLogitsLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)

    avg_proto_loss = 0
    avg_heatmap_loss = 0
    avg_total_loss = 0
    avg_acc = 0

    for i, ((inp,mask),_) in enumerate(train_loader):

        iter_counter += 1
        mask = mask[:way*batch_size]
        
        optimizer.zero_grad()
        
        log_prediction = model.forward_class(inp[way*batch_size:].cuda())
        loss_proto = criterion(log_prediction,target)
        loss_proto.backward()

        heatmap_logits = model.forward_part(inp[:batch_size*way].cuda())
        loss_heatmap = alpha*criterion_part(heatmap_logits,mask.cuda())
        loss_heatmap.backward()

        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        loss = loss_proto+loss_heatmap
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/test_shot/way

        avg_acc += acc
        avg_total_loss += loss.item()
        avg_proto_loss += loss_proto.item()
        avg_heatmap_loss += (loss_heatmap/alpha).item()

        if iter_counter%1000==0:
            model.eval()
            util.visualize(model,writer,iter_counter,inp[:9].cuda(),mask[:9])
            model.train()

    avg_total_loss = avg_total_loss/(i+1)
    avg_proto_loss = avg_proto_loss/(i+1)
    avg_heatmap_loss = avg_heatmap_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('total_loss',avg_total_loss,iter_counter)
    writer.add_scalar('proto_loss',avg_proto_loss,iter_counter)
    writer.add_scalar('heatmap_loss',avg_heatmap_loss,iter_counter)

    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc



def bbN_train(train_loader,model,
    optimizer,writer,iter_counter,alpha):

    test_shot = model.shots[-1]
    way = model.way

    target = torch.LongTensor([i//test_shot for i in range(test_shot*way)]).cuda()
    criterion = NLLLoss().cuda()
    criterion_local = BCELoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)

    avg_proto_loss = 0
    avg_heatmap_loss = 0
    avg_total_loss = 0
    avg_acc = 0

    for i, ((inp,mask),_) in enumerate(train_loader):

        iter_counter += 1      
        inp = inp.cuda()
        mask = mask.cuda()
        mask = torch.cat((mask,1.0-mask),1)

        if iter_counter%1000==0:
            model.eval()
            util.visualize(model,writer,iter_counter,inp[:9],mask[:9])
            model.train()
        
        log_prediction,heatmap = model(inp,mask)

        loss_heatmap = criterion_local(heatmap,mask)
        loss_proto = criterion(log_prediction,target)
        loss = alpha*loss_heatmap+loss_proto
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/test_shot/way

        avg_acc += acc
        avg_total_loss += loss.item()
        avg_proto_loss += loss_proto.item()
        avg_heatmap_loss += loss_heatmap.item()

    avg_total_loss = avg_total_loss/(i+1)
    avg_proto_loss = avg_proto_loss/(i+1)
    avg_heatmap_loss = avg_heatmap_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('total_loss',avg_total_loss,iter_counter)
    writer.add_scalar('proto_loss',avg_proto_loss,iter_counter)
    writer.add_scalar('heatmap_loss',avg_heatmap_loss,iter_counter)

    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc


def fgvc_PN_train(train_loader,oid_loader,model,
    optimizer,writer,iter_counter,alpha):

    way = model.way
    shots = model.shots

    test_shot = shots[-1]
    target = torch.LongTensor([i//test_shot for i in range(test_shot*way)]).cuda()
    criterion = NLLLoss().cuda()
    criterion_part = BCEWithLogitsLoss().cuda()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)

    avg_proto_loss = 0
    avg_heatmap_loss = 0
    avg_total_loss = 0
    avg_acc = 0

    for i, ((inp,_),(oid_img,mask)) in enumerate(zip(train_loader,oid_loader)):

        iter_counter += 1

        optimizer.zero_grad()
        
        log_prediction = model.forward_class(inp.cuda())
        loss_proto = criterion(log_prediction,target)
        loss_proto.backward()

        heatmap_logits = model.forward_part(oid_img.cuda())
        loss_heatmap = alpha*criterion_part(heatmap_logits,mask.cuda())
        loss_heatmap.backward()

        optimizer.step()  

        _,max_index = torch.max(log_prediction,1)
        loss = loss_proto+loss_heatmap
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/test_shot/way

        avg_acc += acc
        avg_total_loss += loss.item()
        avg_proto_loss += loss_proto.item()
        avg_heatmap_loss += (loss_heatmap/alpha).item()

        if iter_counter%1000==0:
            model.eval()
            util.visualize(model,writer,iter_counter,oid_img[:9].cuda(),mask[:9])
            model.train()

    avg_total_loss = avg_total_loss/(i+1)
    avg_proto_loss = avg_proto_loss/(i+1)
    avg_heatmap_loss = avg_heatmap_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('total_loss',avg_total_loss,iter_counter)
    writer.add_scalar('proto_loss',avg_proto_loss,iter_counter)
    writer.add_scalar('heatmap_loss',avg_heatmap_loss,iter_counter)

    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc
