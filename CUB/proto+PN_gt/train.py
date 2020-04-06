import sys
import torch
import numpy as np
from functools import partial
sys.path.append('../../')
from utils import proto_train,proto_eval,networks,dataloader,util

args,name = util.train_parser()

pm = util.Path_Manager('../../dataset/cub_fewshot',args=args)

config = util.Config(args,name,
                     train_annot='part',
                     eval_annot='part')

train_loader = dataloader.meta_train_dataloader(data_path=pm.support,
                                                shots=config.shots,
                                                way=config.way,
                                                annot=config.train_annot,
                                                annot_path=pm.annot_path)

model = networks.Proto_PN_gt(num_part=args.num_part,
                             way=config.way,
                             shots=config.shots,
                             resnet=args.resnet)

model.cuda()

train_func = partial(proto_train.default_train,train_loader=train_loader)
eval_func = proto_eval.default_eval

tm = util.Train_Manager(args,pm,config,
                        train_func=train_func,
                        eval_func=eval_func)

tm.train(model)
proto_eval.eval_test(model,pm,config)