import os
import torch
import torch.optim as optim
import logging
import numpy as np
import argparse
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from . import dataloader


def mkdir(path):
    
    if os.path.exists(path): 
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


def get_logger(filename):

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename,"w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train_parser():

    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt",help="optimizer",choices=['adam','sgd'])
    parser.add_argument("--lr",help="initial learning rate",type=float)
    parser.add_argument("--gamma",help="learning rate cut scalar",type=float,default=0.1)
    parser.add_argument("--epoch",help="number of epochs before lr is cut by gamma",type=int)
    parser.add_argument("--stage",help="number lr stages",type=int)
    parser.add_argument("--weight_decay",help="weight decay for optimizer",type=float)
    parser.add_argument("--gpu",help="gpu device",type=int,default=0)
    parser.add_argument("--seed",help="random seed",type=int,default=42)
    parser.add_argument("--val_epoch",help="number of epochs before eval on val",type=int,default=20)
    parser.add_argument("--resnet", help="whether use resnet18 as backbone or not",action="store_true")
    
    ## PN model related hyper-parameters
    parser.add_argument("--alpha",help="scalar for pose loss",type=int)
    parser.add_argument("--num_part",help="number of parts",type=int)
    parser.add_argument("--percent", help="percent of base images with part annotation",type=float)
    
    ## shared optional
    parser.add_argument("--batch_size",help="batch size",type=int)
    parser.add_argument("--load_path",help="load path for dynamic/transfer models",type=str)

    args = parser.parse_args()

    if args.resnet:
        name = 'ResNet18'
    else:
        name = 'Conv4'

    return args,name


def get_opt(model,args):
    
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.epoch,gamma=args.gamma)

    return optimizer,scheduler


def eval(acc_list):

    mean = np.mean(acc_list)
    interval = 1.96*np.sqrt(np.var(acc_list)/len(acc_list))

    return mean,interval


def visualize(model,writer,iter_counter,inp,mask):
    with torch.no_grad():
        heat_map = model.get_heatmap(inp)
        for j in range(model.num_part):
            pred_part = make_grid(heat_map[:,j,:,:].unsqueeze(1),nrow=3)
            writer.add_image('part_'+str(j)+'/pre',pred_part,iter_counter)
            gt_part = make_grid(mask[:,j,:,:].unsqueeze(1),nrow=3)
            writer.add_image('part_'+str(j)+'/gt',gt_part,iter_counter)


class Path_Manager:
    
    def __init__(self,data_path,args,oid_path=None):
        
        if args.resnet:
            res = 224
        else:
            res = 84

        data_path = os.path.join(os.path.abspath(data_path),'res_'+str(res))

        self.support = os.path.join(data_path,'support')
        self.val_refer = os.path.join(data_path,'val/refer')
        self.val_query = os.path.join(data_path,'val/query')
        self.test_refer = os.path.join(data_path,'test/refer')
        self.test_query = os.path.join(data_path,'test/query')
        self.k_shot = os.path.join(data_path,'eval_k_shot')
        self.class_acc = False

        if oid_path is not None:
            self.oid = os.path.join(os.path.abspath(oid_path),'res_'+str(res))
        else:
            self.annot_path = os.path.join(data_path,'path2annot.pth')
            self.k_shot_annot_path = os.path.join(data_path,'path2annot_eval_k_shot.pth')




class Path_Manager_NA:

    def __init__(self,data_path,args):
        
        if args.resnet:
            res = 224
        else:
            res = 84
        
        data_path = os.path.join(os.path.abspath(data_path),'res_'+str(res))
        self.test_refer = os.path.join(data_path,'refer')
        self.test_query = os.path.join(data_path,'query')
        self.annot_path = os.path.join(data_path,'path2annot.pth')
        self.class_acc = True


class Config:

    def __init__(self,args,name,
                 way=20,
                 shots=[5,15],
                 train_annot=None,
                 eval_annot=None,
                 suffix=None):

        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        torch.cuda.set_device(args.gpu)

        if suffix is not None:
            name = "%s-%s"%(name,suffix)
        self.logger = get_logger('%s.log' % (name))
        self.save_path = 'model_%s.pth' % (name)
        self.writer = SummaryWriter('log_%s' % (name))

        self.way = way
        self.shots = shots
        self.train_annot = train_annot
        self.eval_annot = eval_annot

        self.logger.info('display all the hyper-parameters in args:')
        for arg in vars(args):
            value = getattr(args,arg)
            if value is not None:
                self.logger.info('%s: %s' % (str(arg),str(value)))
        self.logger.info('------------------------')



class Train_Manager:

    def __init__(self,args,pm,config,
                 train_func,
                 eval_func=None):

        self.pm = pm
        self.config = config
        self.args = args
        self.train_func = train_func
        self.eval_func = eval_func
    
    def train(self,model):

        pm = self.pm
        args = self.args
        config = self.config
        train_func = self.train_func
        eval_func = self.eval_func

        optimizer,scheduler = get_opt(model,args)
        validation = eval_func is not None

        if validation:
            val_refer_loader = dataloader.eval_dataloader(pm.val_refer,
                annot=config.eval_annot,annot_path=pm.annot_path)
            val_query_loader = dataloader.eval_dataloader(pm.val_query,
                annot=config.eval_annot,annot_path=pm.annot_path)
            best_val_acc = 0
            best_epoch = 0

        writer = config.writer
        save_path = config.save_path
        logger = config.logger

        self.set_train_mode(model) 

        iter_counter = 0
        total_epoch = args.epoch*args.stage

        logger.info("start training!")

        for e in range(total_epoch):

            iter_counter,train_acc = train_func(model=model,
                                                optimizer=optimizer,
                                                writer=writer,
                                                iter_counter=iter_counter)

            if (e+1)%args.val_epoch==0:

                logger.info("")
                logger.info("epoch %d/%d, iter %d:" % (e+1,total_epoch,iter_counter))
                logger.info("train_acc: %.3f" % (train_acc))

                if validation:

                    model.eval()                    
                    with torch.no_grad():
                        val_acc = eval_func(val_refer_loader,val_query_loader,model)
                        writer.add_scalar('val_acc',val_acc,iter_counter)

                    logger.info("val_acc: %.3f" % (val_acc))
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = e+1
                        torch.save(model.state_dict(),save_path)
                        logger.info('BEST!')

                    self.set_train_mode(model)

            scheduler.step()

        logger.info('training finished!')

        if validation:
            logger.info('------------------------')
            logger.info(('the best epoch is %d/%d') % (best_epoch,total_epoch))
            logger.info(('the best val acc is %.3f') % (best_val_acc))

        else:
            torch.save(model.state_dict(),save_path)

    def set_train_mode(self,model):
        model.train()




class TM_dynamic_stage_2(Train_Manager):
    
    def set_train_mode(self,model):

        model.train()

        model.feature_extractor.eval()
        for param in model.feature_extractor.parameters():
            param.requires_grad = False



class TM_dynamic_PN_stage_2(Train_Manager):

    def set_train_mode(self,model):

        model.train()

        model.PN_Model.eval()
        for param in model.PN_Model.parameters():
            param.requires_grad = False



class TM_transfer_finetune(Train_Manager):

    def set_train_mode(self,model):

        model.feature_extractor.eval()
        
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

        model.linear_classifier.train()



class TM_transfer_PN_finetune(Train_Manager):

    def set_train_mode(self,model):

        model.shared_layers.eval()
        for param in model.shared_layers.parameters():
            param.requires_grad = False

        model.class_branch.eval()
        for param in model.class_branch.parameters():
            param.requires_grad = False

        model.part_branch.eval()
        for param in model.part_branch.parameters():
            param.requires_grad = False

        model.linear_classifier.train()


