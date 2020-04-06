import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from . import util,dataloader

def default_eval(refer_loader,query_loader,model,class_acc=False):

    class_weight = get_class_weight(refer_loader,model)

    acc = get_prediction(query_loader,model,class_weight,class_acc)

    return acc



def get_class_weight(loader,model):

    dim = model.dim
    way = len(loader.dataset.classes)
    class_weight = torch.zeros(way,dim).cuda()

    for i, (inp,target) in enumerate(loader):

        current_class_id = target[0]

        with torch.no_grad():

            if isinstance(inp,list):
                (image_inp,mask) = inp
                image_inp = image_inp.cuda()
                mask = mask.cuda()
                feature_vector = model.get_feature_vector(image_inp,mask)

            elif isinstance(inp,torch.Tensor):
                inp = inp.cuda()
                feature_vector = model.get_feature_vector(inp)

            class_weight[current_class_id] = model.get_single_class_weight(feature_vector)

    return class_weight




def get_prediction(loader,model,class_weight,class_acc):

    data_source = loader.dataset

    way = len(data_source.classes)

    correct_count = torch.zeros(way).cuda()

    counts = torch.zeros(way).cuda()
    
    for class_id in data_source.targets:
        counts[class_id] += 1

    for i, (inp,target) in enumerate(loader):

        current_class_id = target[0]
        target = target.cuda()
        batch_size = target.size(0)

        if isinstance(inp,list):
            (image_inp,mask) = inp
                
            image_inp = image_inp.cuda()
            mask = mask.cuda()
            feature_vector = model.get_feature_vector(image_inp,mask)

        elif isinstance(inp,torch.Tensor):
            inp = inp.cuda()
            feature_vector = model.get_feature_vector(inp)
        
        prediction = model.get_prediction(feature_vector,class_weight)

        _, top1_pred = prediction.topk(1)

        correct_count[current_class_id] = torch.sum(torch.eq(top1_pred,target.view(batch_size,1)))

    acc =  (torch.sum(correct_count)/torch.sum(counts)).item()*100

    if not class_acc:
        return acc
    else:
        class_acc = torch.mean(correct_count/counts).item()*100
        return acc,class_acc




def k_shot_eval(eval_loader,model,way,shot):
    
    test_shot = 16
    target = torch.LongTensor([i//test_shot for i in range(test_shot*way)]).cuda()

    acc_list = []

    for i, (inp,_) in enumerate(eval_loader):

        if isinstance(inp,list):
            (image_inp,mask) = inp
            image_inp = image_inp.cuda()
            mask = mask.cuda()
            feature_vector = model.get_feature_vector(image_inp,mask)

        elif isinstance(inp,torch.Tensor):
            inp = inp.cuda()
            feature_vector = model.get_feature_vector(inp)
           
        max_index = model.eval_k_shot(feature_vector,way,shot)

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

