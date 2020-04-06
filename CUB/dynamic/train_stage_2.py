import sys
import torch
import numpy as np
from functools import partial
sys.path.append('../../')
from utils import dynamic_train,dynamic_eval,networks,dataloader,util

args,name = util.train_parser()

pm = util.Path_Manager('../../dataset/cub_fewshot',args=args)

config = util.Config(args=args,
                     name=name,
                     suffix='stage_2',
                     shots=[20])

train_loader = dataloader.meta_train_dataloader(data_path=pm.support,
                                                way=config.way,
                                                shots=config.shots)

num_class = len(train_loader.dataset.classes)

model = networks.Dynamic(num_class=num_class,
                         way=config.way,
                         shots=config.shots,
                         resnet=args.resnet)
model.cuda()

model.load_state_dict(torch.load(args.load_path))

train_func = partial(dynamic_train.train_stage_2,train_loader=train_loader)
eval_func = dynamic_eval.default_eval

tm = util.TM_dynamic_stage_2(args,pm,config,
                             train_func=train_func,
                             eval_func=eval_func)

tm.train(model)

pm_na = util.Path_Manager_NA('../../dataset/na_fewshot',args=args)
dynamic_eval.eval_test(model,pm,config,pm_na=pm_na)