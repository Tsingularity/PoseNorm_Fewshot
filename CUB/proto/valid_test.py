import sys
import os
import torch
import numpy as np
from functools import partial
sys.path.append('../../')
from utils import proto_train,proto_eval,networks,dataloader,util

res = 84
data_path = '/home/lt453/PN_pre/dataset/cub_fewshot/'
model_path = '/home/lt453/pose_normalization/proto_seed/model_sgd-lr_1e-01-gamma_1e-01-epoch_400-stage_2-decay_5e-04-seed_42.pth'

gpu = 0
torch.cuda.set_device(gpu)

data_path = os.path.join(data_path,'res_%d'%(res))
test_refer = os.path.join(data_path,'test/refer')
test_query = os.path.join(data_path,'test/query')
test_refer_loader = dataloader.eval_dataloader(test_refer)
test_query_loader = dataloader.eval_dataloader(test_query)
eval_k_shot_path = os.path.join(data_path,'eval_k_shot')

na_path = os.path.join('/home/lt453/PN_pre/dataset/na_fewshot/','res_%d'%(res))
na_test_refer = os.path.join(na_path,'refer')
na_test_query = os.path.join(na_path,'query')

model = networks.Proto()
model.cuda()

with torch.no_grad():
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    refer_loader = dataloader.eval_dataloader(test_refer)
    query_loader = dataloader.eval_dataloader(test_query)

    test_acc = proto_eval.default_eval(refer_loader,query_loader,model=model)
    print(('the final test acc is %.3f') % (test_acc))

    way = len(refer_loader.dataset.classes)
    for shot in [1,5]:
        eval_loader = dataloader.eval_k_shot_dataloader(eval_k_shot_path,way=way,shot=shot)
        mean,interval = proto_eval.k_shot_eval(eval_loader,model,way,shot)
        print('%d-way-%d-shot acc: %.2f\t%.2f'%(way,shot,mean,interval))


    print('evaluating on NA:')
    
    refer_loader = dataloader.eval_dataloader(na_test_refer)
    query_loader = dataloader.eval_dataloader(na_test_query)

    mean_acc,class_acc = proto_eval.default_eval(refer_loader,query_loader,
        model=model,class_acc=True)
    
    print(('mean_acc is %.3f') % (mean_acc))
    print(('class_acc is %.3f') % (class_acc))