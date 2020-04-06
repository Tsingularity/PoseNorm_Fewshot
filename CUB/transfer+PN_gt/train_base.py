import sys
import torch
import numpy as np
from functools import partial
sys.path.append('../../')
from utils import transfer_train,networks,dataloader,util

args,name = util.train_parser()

pm = util.Path_Manager('../../dataset/cub_fewshot',args=args)

config = util.Config(args=args,
                     name=name,
                     suffix='base',
                     train_annot='part')

train_loader = dataloader.normal_train_dataloader(data_path=pm.support,
                                                  batch_size=args.batch_size,
                                                  annot=config.train_annot,
                                                  annot_path=pm.annot_path)

model = networks.Transfer_PN_gt(num_part=args.num_part,
                                resnet=args.resnet)

model.cuda()

train_func = partial(transfer_train.default_train,train_loader=train_loader)

tm = util.Train_Manager(args,pm,config,
                        train_func=train_func)

tm.train(model)