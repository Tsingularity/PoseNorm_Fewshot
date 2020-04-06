import sys
import torch
import numpy as np
from functools import partial
sys.path.append('../../')
from utils import dynamic_train,networks,dataloader,util

args,name = util.train_parser()

pm = util.Path_Manager('../../dataset/cub_fewshot',args=args)

config = util.Config(args=args,
                     name=name,
                     suffix='stage_1')

train_loader = dataloader.normal_train_dataloader(data_path=pm.support,
                                                  batch_size=args.batch_size)

num_class = len(train_loader.dataset.classes)

model = networks.Dynamic(num_class=num_class,resnet=args.resnet)

model.cuda()

train_func = partial(dynamic_train.train_stage_1,train_loader=train_loader)

tm = util.Train_Manager(args,pm,config,
                train_func=train_func)

tm.train(model)