import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from . import util,dataloader

def default_eval(refer_loader,query_loader,model,class_acc=False):
    
    fb_vector = None
    if hasattr(model,'get_fb_vector'):
        fb_vector = get_fb_vector(refer_loader,model)

    centroid = get_class_centroid(refer_loader,model,fb_vector)

    acc = get_prediction(query_loader,model,centroid,fb_vector,class_acc)

    return acc


def get_class_centroid(loader,model,fb_vector=None):
    
    way = len(loader.dataset.classes)
    dim = model.dim

    centroid = torch.zeros(way,dim).cuda()

    for i, (inp,target) in enumerate(loader):

        current_class_id = target[0]

        if fb_vector is not None:
            (image,_) = inp
            image = image.cuda()
            vectors = model.get_feature_vector(image,fb_vector)
        
        elif isinstance(inp,list):
            (image,mask) = inp
            image = image.cuda()
            mask = mask.cuda()
            vectors = model.get_feature_vector(image,mask)

        elif isinstance(inp,torch.Tensor):
            inp = inp.cuda()
            vectors = model.get_feature_vector(inp)

        centroid[current_class_id] = vectors.mean(0).view(dim)

    return centroid


def get_prediction(loader,model,centroid,fb_vector=None,class_acc=False):

    data_source = loader.dataset
    centroid = centroid.unsqueeze(0)

    way = len(data_source.classes)

    correct_count = torch.zeros(way).cuda()

    counts = torch.zeros(way).cuda()
    
    for class_id in data_source.targets:
        counts[class_id] += 1

    for i, (inp,target) in enumerate(loader):

        current_class_id = target[0]
        batch_size = target.size(0)
        target = target.cuda()

        if fb_vector is not None:
            (image,mask) = inp
            image = image.cuda()
            out = model.get_feature_vector(image,fb_vector)

        elif isinstance(inp,list):
            (image,mask) = inp
            image = image.cuda()
            mask = mask.cuda()
            out = model.get_feature_vector(image,mask)

        elif isinstance(inp,torch.Tensor):
            inp = inp.cuda()
            out = model.get_feature_vector(inp)

        out = out.unsqueeze(1)
        neg_l2_distance = torch.sum((centroid-out)**2,2).neg().view(batch_size,way)

        _, top1_pred = neg_l2_distance.topk(1)

        correct_count[current_class_id] = torch.sum(torch.eq(top1_pred,target.view(batch_size,1)))

    acc =  (torch.sum(correct_count)/torch.sum(counts)).item()*100

    if not class_acc:
        return acc
    else:
        class_acc = torch.mean(correct_count/counts).item()*100
        return acc,class_acc



def get_fb_vector(loader,model):

    num_channel = model.num_channel
    sum_fb_vector = torch.zeros(num_channel,2).cuda()
    total_num = 0

    for i,((inp,mask),class_id) in enumerate(loader):

        total_num += inp.size(0)
        inp=inp.cuda()
        mask=mask.cuda()
   
        fb_vector = model.get_fb_vector(inp,mask)
        sum_fb_vector += fb_vector.sum(0)

    fb_vector = sum_fb_vector/total_num

    return fb_vector


def k_shot_eval(eval_loader,model,way,shot):
    
    test_shot = 16
    target = torch.LongTensor([i//test_shot for i in range(test_shot*way)]).cuda()

    acc_list = []

    for i, (inp,_) in enumerate(eval_loader):

        if isinstance(inp,list):
            (image_inp,mask) = inp
            image_inp = image_inp.cuda()
            mask = mask.cuda()
            max_index = model.eval_k_shot(image_inp,mask,way,shot)

        elif isinstance(inp,torch.Tensor):
            inp = inp.cuda()
            max_index = model.eval_k_shot(inp,way,shot)

        acc = 100*torch.sum(torch.eq(max_index,target)).item()/test_shot/way
        acc_list.append(acc)

    mean,interval = util.eval(acc_list)

    return mean,interval


def eval_test(model,pm,config,pm_na=None):

    logger = config.logger
    annot = config.eval_annot

    logger.info('------------------------')
    logger.info('evaluating:')

    with torch.no_grad():
        
        model.load_state_dict(torch.load(config.save_path))
        model.eval()

        refer_loader = dataloader.eval_dataloader(pm.test_refer,
            annot=annot,annot_path=pm.annot_path)
        query_loader = dataloader.eval_dataloader(pm.test_query,
            annot=annot,annot_path=pm.annot_path)

        test_acc = default_eval(refer_loader,query_loader,model=model)
        logger.info(('the final test acc is %.3f') % (test_acc))

        way = len(refer_loader.dataset.classes)
        for shot in [1,5]:
            eval_loader = dataloader.eval_k_shot_dataloader(pm.k_shot,
                way=way,shot=shot,annot=annot,annot_path=pm.k_shot_annot_path)
            mean,interval = k_shot_eval(eval_loader,model,way,shot)
            logger.info('%d-way-%d-shot acc: %.2f\t%.2f'%(way,shot,mean,interval))

        if pm_na is not None:

            logger.info('------------------------')
            logger.info('evaluating on NA:')
            
            refer_loader = dataloader.eval_dataloader(pm_na.test_refer,
                annot=annot,annot_path=pm_na.annot_path)
            query_loader = dataloader.eval_dataloader(pm_na.test_query,
                annot=annot,annot_path=pm_na.annot_path)

            mean_acc,class_acc = default_eval(refer_loader,query_loader,
                model=model,class_acc=True)
            
            logger.info(('mean_acc is %.3f') % (mean_acc))
            logger.info(('class_acc is %.3f') % (class_acc))

