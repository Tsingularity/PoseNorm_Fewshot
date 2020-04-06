import sys
import torch
import numpy as np
from functools import partial
sys.path.append('../../')
from utils import proto_train,proto_eval,networks,dataloader,util

args,name = util.train_parser()

pm = util.Path_Manager('../../dataset/fgvc_fewshot',args=args)

config = util.Config(args,name)

train_loader = dataloader.meta_train_dataloader(data_path=pm.support,
                                                shots=config.shots,
                                                way=config.way)

if args.resnet:
    res=224
else:
    res=84
oid_path = '../../dataset/oid_fewshot/res_%d'%(res)
oid_loader = dataloader.oid_dataloader(oid_path,args.batch_size)

model = networks.Proto_PN_less_annot(num_part=args.num_part,
                                     way=config.way,
                                     shots=config.shots,
                                     resnet=args.resnet)

model.cuda()

train_func = partial(proto_train.fgvc_PN_train,
                     train_loader=train_loader,
                     oid_loader=oid_loader,
                     alpha=args.alpha)
eval_func = proto_eval.default_eval

tm = util.Train_Manager(args,pm,config,
                        train_func=train_func,
                        eval_func=eval_func)

tm.train(model)

proto_eval.eval_test(model,pm,config)