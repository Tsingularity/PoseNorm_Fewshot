import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from . import models

EPS=0.00001


def feature_map2vec(feature_map,mask):

    batch_size = feature_map.size(0)
    num_channel = feature_map.size(1)
    num_part = mask.size(1)

    feature_map = feature_map.unsqueeze(2)
    sum_of_weight = mask.view(batch_size,num_part,-1).sum(-1)+EPS
    mask = mask.unsqueeze(1)

    vec = (feature_map*mask).view(batch_size,num_channel,num_part,-1).sum(-1)
    vec = vec/sum_of_weight.unsqueeze(1)
    vec = vec.view(batch_size,num_channel*num_part)

    return vec



class Proto(models.Proto_Model):

    def get_feature_vector(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        feature_vector = F.avg_pool2d(input=feature_map,kernel_size=feature_map.size(-1))
        feature_vector = feature_vector.view(batch_size,-1)

        return feature_vector


class Proto_BP(models.Proto_Model):

    def __init__(self,way=None,shots=None,resnet=False):
        
        super().__init__(way=way,shots=shots,resnet=resnet)
        self.dim = self.num_channel**2

    def get_feature_vector(self,inp):

        feature_map = self.feature_extractor(inp)
        feature_vector = self.covariance_pool(feature_map)
        return feature_vector

    def covariance_pool(self,inp):

        batch_size = inp.size(0)
        channel = inp.size(1)
        height = inp.size(2)
        width = inp.size(3)

        out = inp.view(batch_size,channel,height*width)
        out = torch.bmm(out,torch.transpose(out,1,2))/(height*width)
        out = out.view(batch_size,channel**2)
        out = out.sign().float()*(out.abs()+EPS).sqrt()

        return out


class Proto_FSL(models.Proto_Model):

    def __init__(self,way=None,shots=None,resnet=False):
        
        super().__init__(way=way,shots=shots,resnet=resnet)
        self.dim = self.num_channel*2


    def get_fb_vector(self,inp,mask):

        batch_size = inp.size(0)
        num_channel = self.num_channel

        feature_map = self.feature_extractor(inp).unsqueeze(2)
        mask = torch.cat([mask,1-mask],dim=1).view(batch_size,1,2,mask.size(2),mask.size(3))
        sum_of_weight = mask.view(*mask.size()[:-2],-1).sum(-1)+EPS
        fb_vector = (feature_map*mask).view(batch_size,num_channel,2,-1).sum(-1)/sum_of_weight

        return fb_vector


    def get_feature_vector(self,inp,fb_vector):

        feature_map = self.feature_extractor(inp)
        mask = self.feature_map2mask(feature_map,fb_vector)
        feature_vector = feature_map2vec(feature_map,mask)

        return feature_vector

    def feature_map2mask(self,feature_map,fb_vector):

        num_channel = self.num_channel

        feature_map = feature_map.unsqueeze(2)
        fb_vector = fb_vector.view(1,num_channel,2,1,1)

        mask = torch.sum((feature_map-fb_vector)**2,1).neg() 
        mask = F.softmax(mask,dim=1)

        return mask


    def eval_k_shot(self,inp,mask,way,shot):

        fb_vector = self.get_fb_vector(inp[:way*shot],mask[:way*shot]).mean(0)
        feature_vector = self.get_feature_vector(inp,fb_vector)

        max_index = models.proto_eval_k_shot(feature_vector,
            way = way,
            shot = shot,
            dim = self.dim)

        return max_index


    def forward(self,inp,mask):
        
        feature_map = self.feature_extractor(inp)

        refer_shot = self.shots[0]
        train_shot = self.shots[1]
        test_shot = self.shots[2]

        way = self.way
        num_channel = self.num_channel
        refer_batch_size = way*refer_shot

        refer_mask = mask[:refer_batch_size]
        refer_feature_map = feature_map[:refer_batch_size]
        refer_mask = torch.cat([refer_mask,1-refer_mask],dim=1).unsqueeze(1)
        sum_of_weight = refer_mask.view(*refer_mask.size()[:-2],-1).sum(-1)+EPS

        fb_vector = (refer_feature_map.unsqueeze(2)*refer_mask).view(refer_batch_size,num_channel,2,-1).sum(-1)/sum_of_weight
        fb_vector = fb_vector.mean(0)

        feature_map = feature_map[refer_batch_size:]
        fb_mask = self.feature_map2mask(feature_map,fb_vector)
        feature_vector = feature_map2vec(feature_map,fb_mask)

        log_prediction = models.proto_forward_log_pred(feature_vector,
            train_shot = train_shot,
            test_shot = test_shot,
            dim = self.dim,
            way = way)

        return log_prediction


class Proto_bbN(models.PN_Model):

    def __init__(self,num_part=2,way=None,shots=None,resnet=False):
        
        super().__init__(num_part=num_part,resnet=resnet)
        self.shots = shots
        self.way = way

    def get_feature_vector(self,inp):

        temp = self.shared_layers(inp)
        feature_map = self.class_branch(temp)
        mask = nn.Softmax(dim=1)(self.part_branch(temp))

        vec = feature_map2vec(feature_map,mask)

        return vec

    def get_heatmap(self,inp):

        logits = self.part_branch(self.shared_layers(inp))
        heat_map = nn.Softmax(dim=1)(logits)
        return heat_map


    def forward(self,inp,mask):

        temp = self.shared_layers(inp)
        feature_map = self.class_branch(temp)
        heatmap = nn.Softmax(dim=1)(self.part_branch(temp))

        feature_vector = feature_map2vec(feature_map,mask)

        log_prediction = models.proto_forward_log_pred(feature_vector,
            train_shot = self.shots[0],
            test_shot = self.shots[1],
            dim = self.dim,
            way = self.way)

        return log_prediction,heatmap



class Proto_MT(models.PN_Model):

    def __init__(self,num_part=15,way=None,shots=None,resnet=False):
        
        super().__init__(num_part=num_part,resnet=resnet)
        self.shots = shots
        self.way = way
        self.dim = self.num_channel

    def get_feature_vector(self,inp):

        batch_size = inp.size(0)
        temp = self.shared_layers(inp)
        feature_map = self.class_branch(temp)

        feature_vector = F.avg_pool2d(input=feature_map,kernel_size=feature_map.size(-1))
        feature_vector = feature_vector.view(batch_size,-1)

        return feature_vector


    def forward(self,inp,mask):

        temp = self.shared_layers(inp)
        feature_map = self.class_branch(temp)
        heatmap_logits = self.part_branch(temp)

        feature_vector = F.avg_pool2d(input=feature_map,kernel_size=feature_map.size(-1))
        feature_vector = feature_vector.view(feature_vector.size(0),-1)

        log_prediction = models.proto_forward_log_pred(feature_vector,
            train_shot = self.shots[0],
            test_shot = self.shots[1],
            dim = self.dim,
            way = self.way)

        return log_prediction,heatmap_logits



class Proto_uPN(models.Proto_Model):

    def __init__(self,num_part=15,way=None,shots=None,resnet=False):
        
        super().__init__(way=way,shots=shots,resnet=resnet)
        num_channel = self.num_channel
        self.dim = num_channel*num_part
        self.num_part = num_part
        self.part_vector = nn.Parameter(torch.randn(1,num_channel,num_part,1,1))

    def get_feature_vector(self,inp):

        num_channel = self.num_channel
        num_part = self.num_part
        dim = self.dim

        batch_size = inp.size(0)

        feature_map = self.feature_extractor(inp)
        fm_size = feature_map.size(-1)
        feature_map = feature_map.unsqueeze(2)
  
        mask = torch.sum((feature_map-self.part_vector)**2,1).neg().view(batch_size,num_part,-1)
        mask = F.softmax(mask,dim=2).view(batch_size,1,num_part,fm_size,fm_size)

        feature_vector = (feature_map*mask).view(batch_size,num_channel,num_part,-1).sum(-1)
        feature_vector = feature_vector.view(batch_size,dim)

        return feature_vector

    def get_heatmap(self,inp):

        num_part = self.num_part

        batch_size = inp.size(0)

        feature_map = self.feature_extractor(inp)
        fm_size = feature_map.size(-1)
        feature_map = feature_map.unsqueeze(2)
  
        mask = torch.sum((feature_map-self.part_vector)**2,1).neg().view(batch_size,num_part,-1)
        mask = F.softmax(mask,dim=2).view(batch_size,num_part,fm_size,fm_size)

        return mask



class Proto_PN(models.PN_Model):

    def __init__(self,num_part=15,way=None,shots=None,resnet=False):
        
        super().__init__(num_part=num_part,resnet=resnet)
        self.shots = shots
        self.way = way

    def get_feature_vector(self,inp):

        temp = self.shared_layers(inp)
        feature_map = self.class_branch(temp)
        mask = nn.Sigmoid()(self.part_branch(temp))

        vec = feature_map2vec(feature_map,mask)

        return vec

    def forward(self,inp,mask):

        temp = self.shared_layers(inp)
        feature_map = self.class_branch(temp)
        heatmap_logits = self.part_branch(temp)

        feature_vector = feature_map2vec(feature_map,mask)

        log_prediction = models.proto_forward_log_pred(feature_vector,
            train_shot = self.shots[0],
            test_shot = self.shots[1],
            dim = self.dim,
            way = self.way)

        return log_prediction,heatmap_logits



class Proto_PN_less_annot(Proto_PN):

    def forward_class(self,inp):

        feature_vector = self.get_feature_vector(inp)

        log_prediction = models.proto_forward_log_pred(feature_vector,
            train_shot = self.shots[0],
            test_shot = self.shots[1],
            dim = self.dim,
            way = self.way)

        return log_prediction

    def forward_part(self,inp):

        temp = self.shared_layers(inp)
        heatmap_logits = self.part_branch(temp)

        return heatmap_logits




class Proto_PN_gt(models.Proto_Model):

    def __init__(self,num_part=15,way=None,shots=None,resnet=False):
        
        super().__init__(way=way,shots=shots,resnet=resnet)
        self.dim = self.num_channel*num_part
        self.num_part = num_part

    def get_feature_vector(self,inp,mask):

        batch_size = inp.size(0)
        num_channel = self.num_channel
        num_part = self.num_part
        dim = self.dim

        feature_map = self.feature_extractor(inp).unsqueeze(2)
        mask = mask.unsqueeze(1)

        feature_vector = (feature_map*mask).view(batch_size,num_channel,num_part,-1).sum(-1)
        feature_vector = feature_vector.view(batch_size,dim)

        return feature_vector

    def eval_k_shot(self,inp,mask,way,shot):

        feature_vector = self.get_feature_vector(inp,mask)
        
        max_index = models.proto_eval_k_shot(feature_vector,
            way = way,
            shot = shot,
            dim = self.dim)

        return max_index

    def forward(self,inp,mask):

        feature_vector = self.get_feature_vector(inp,mask)
        
        log_prediction = models.proto_forward_log_pred(feature_vector,
            train_shot = self.shots[0],
            test_shot = self.shots[1],
            dim = self.dim,
            way = self.way)

        return log_prediction



class Transfer(nn.Module):

    def __init__(self,num_class=100,resnet=False):
        
        super().__init__()
        if resnet:
            num_channel = 32
            self.feature_extractor = models.BackBone_ResNet(num_channel)
        else:
            num_channel = 64
            self.feature_extractor = models.BackBone(num_channel)

        self.linear_classifier = nn.Linear(num_channel,num_class)
        self.num_channel = num_channel
        self.num_class = num_class
        self.dim = num_channel

    def get_feature_vector(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        feature_vector = F.avg_pool2d(input=feature_map,kernel_size=feature_map.size(-1))
        feature_vector = feature_vector.view(batch_size,-1)

        return feature_vector

    def forward(self,inp):
        
        feature_vector = self.get_feature_vector(inp)
        logits = self.linear_classifier(feature_vector)
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction


class Transfer_PN(models.PN_Model):

    def __init__(self,num_class=100,num_part=15,resnet=False):
        
        super().__init__(num_part=num_part,resnet=resnet)
        self.linear_classifier = nn.Linear(self.dim,num_class)
        self.num_class = num_class

    def forward(self,inp,mask=None):
        
        batch_size = inp.size(0)
        num_channel = self.num_channel
        num_part = self.num_part

        temp = self.shared_layers(inp)
        feature_map = self.class_branch(temp)
        heatmap_logits = self.part_branch(temp)

        is_training = True

        if mask is None:
            mask = nn.Sigmoid()(heatmap_logits)
            is_training = False

        feature_vector = feature_map2vec(feature_map,mask)

        logits = self.linear_classifier(feature_vector)
        log_prediction = F.log_softmax(logits,dim=1)

        if is_training:
            return log_prediction,heatmap_logits
        else:
            return log_prediction


class Transfer_PN_gt(nn.Module):

    def __init__(self,num_class=100,num_part=15,resnet=False):
        
        super().__init__()
        if resnet:
            num_channel = 32
            self.feature_extractor = models.BackBone_ResNet(num_channel)
        else:
            num_channel = 64
            self.feature_extractor = models.BackBone(num_channel)

        self.num_channel = num_channel
        self.num_part = num_part
        self.num_class = num_class
        self.dim = num_channel*num_part
        self.linear_classifier = nn.Linear(self.dim,num_class)

    def get_feature_vector(self,inp,mask):

        batch_size = inp.size(0)
        num_channel = self.num_channel
        num_part = self.num_part
        dim = self.dim

        feature_map = self.feature_extractor(inp).unsqueeze(2)
        mask = mask.unsqueeze(1)

        feature_vector = (feature_map*mask).view(batch_size,num_channel,num_part,-1).sum(-1)
        feature_vector = feature_vector.view(batch_size,dim)

        return feature_vector

    def forward(self,inp,mask):
        
        feature_vector = self.get_feature_vector(inp,mask)
        logits = self.linear_classifier(feature_vector)
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction



class Dynamic(models.Dynamic_Model):

    def __init__(self,num_class=100,resnet=False,way=None,shots=None):

        if resnet:
            num_channel = 32
        else:
            num_channel = 64

        super().__init__(num_class=num_class,dim=num_channel,way=way,shots=shots)

        if resnet:
            self.feature_extractor = models.BackBone_ResNet(num_channel)
        else:
            self.feature_extractor = models.BackBone(num_channel)


    def get_feature_vector(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        feature_vector = F.avg_pool2d(input=feature_map,kernel_size=feature_map.size(-1))
        feature_vector = feature_vector.view(batch_size,-1)

        return feature_vector

    def forward_stage_1(self,inp):

        feature_vector = self.get_feature_vector(inp)
        log_prediction = self.get_prediction(feature_vector,self.weight_base)

        return log_prediction


class Dynamic_PN(models.Dynamic_Model):

    def __init__(self,num_class=100,num_part=15,resnet=False,way=None,shots=None):

        if resnet:
            num_channel = 32
        else:
            num_channel = 64

        super().__init__(num_class=num_class,dim=num_channel*num_part,way=way,shots=shots)

        self.PN_Model = models.PN_Model(num_part=num_part,resnet=resnet)
        self.num_part = num_part

    def get_feature_vector(self,inp):

        temp = self.PN_Model.shared_layers(inp)
        feature_map = self.PN_Model.class_branch(temp)
        mask = nn.Sigmoid()(self.PN_Model.part_branch(temp))

        vec = feature_map2vec(feature_map,mask)

        return vec

    def get_heatmap(self,inp):

        return self.PN_Model.get_heatmap(inp)

    def forward_stage_1(self,inp,mask):

        dim = self.dim
        batch_size = inp.size(0)
        
        temp = self.PN_Model.shared_layers(inp)
        feature_map = self.PN_Model.class_branch(temp)
        heatmap_logits = self.PN_Model.part_branch(temp)

        feature_vector = feature_map2vec(feature_map,mask).view(batch_size,dim)
        log_prediction = self.get_prediction(feature_vector,self.weight_base)

        return log_prediction,heatmap_logits


class Dynamic_PN_gt(models.Dynamic_Model):

    def __init__(self,num_class=100,num_part=15,resnet=False,way=None,shots=None):

        if resnet:
            num_channel = 32
        else:
            num_channel = 64

        super().__init__(num_class=num_class,dim=num_channel*num_part,way=way,shots=shots)

        self.num_part = num_part
        self.num_channel = num_channel

        if resnet:
            self.feature_extractor = models.BackBone_ResNet(num_channel)
        else:
            self.feature_extractor = models.BackBone(num_channel)

    def get_feature_vector(self,inp,mask):

        batch_size = inp.size(0)
        num_channel = self.num_channel
        num_part = self.num_part
        dim = self.dim

        feature_map = self.feature_extractor(inp).unsqueeze(2)
        mask = mask.unsqueeze(1)

        feature_vector = (feature_map*mask).view(batch_size,num_channel,num_part,-1).sum(-1)
        feature_vector = feature_vector.view(batch_size,dim)

        return feature_vector

    def forward_stage_1(self,inp,mask):

        feature_vector = self.get_feature_vector(inp,mask)
        log_prediction = self.get_prediction(feature_vector,self.weight_base)

        return log_prediction