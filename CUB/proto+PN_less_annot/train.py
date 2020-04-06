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
                     suffix="percent_%d-bz_%d"%(args.percent,args.batch_size))

train_loader = dataloader.proto_train_less_annot_dataloader(data_path=pm.support,
                                                            shots=config.shots,
                                                            way=config.way,
                                                            annot_path=pm.annot_path,
                                                            percent=args.percent,
                                                            batch_size=args.batch_size)

model = networks.Proto_PN_less_annot(num_part=args.num_part,
                                     way=config.way,
                                     shots=config.shots,
                                     resnet=args.resnet)

model.cuda()

train_func = partial(proto_train.PN_train_less_annot,
                     train_loader=train_loader,
                     alpha=args.alpha,
                     batch_size=args.batch_size)

eval_func = proto_eval.default_eval

tm = util.Train_Manager(args,pm,config,
                        train_func=train_func,
                        eval_func=eval_func)

tm.train(model)

pm_na = util.Path_Manager_NA('../../dataset/na_fewshot',args=args)
proto_eval.eval_test(model,pm,config,pm_na=pm_na)