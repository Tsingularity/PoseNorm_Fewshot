import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from . import dataloader

def default_eval(loader,model,class_acc=False):

    data_source = loader.dataset

    way = len(data_source.classes)

    correct_count = torch.zeros(way).cuda()

    counts = torch.zeros(way).cuda()
    
    for class_id in data_source.targets:
        counts[class_id] += 1

    with torch.no_grad():

        for i, (inp,target) in enumerate(loader):

            current_class_id = target[0]
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

            _, top1_pred = log_prediction.topk(1)

            correct_count[current_class_id] = torch.sum(torch.eq(top1_pred,target.view(batch_size,1)))

        acc =  (torch.sum(correct_count)/torch.sum(counts)).item()*100

        if not class_acc:
            return acc
        else:
            class_acc = torch.mean(correct_count/counts).item()*100
            return [acc,class_acc]


def eval_test(model,pm,config):

    logger = config.logger
    annot = config.eval_annot

    logger.info('------------------------')
    logger.info('evaluating:')

    with torch.no_grad():

        model.eval()

        query_loader = dataloader.eval_dataloader(pm.test_query,
            annot=annot,annot_path=pm.annot_path)

        test_acc = default_eval(query_loader,
            model=model,class_acc=pm.class_acc)

        if isinstance(test_acc,list):
            mean_acc,class_acc = test_acc            
            logger.info(('mean_acc is %.3f') % (mean_acc))
            logger.info(('class_acc is %.3f') % (class_acc))
        else:
            logger.info(('the final test acc is %.3f') % (test_acc))