import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np

class ConvBlock(nn.Module):
    
    def __init__(self,input_channel,output_channel):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self,inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self,num_channel=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(3,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel))

    def forward(self,inp):

        return self.layers(inp)


class BackBone_ResNet(nn.Module):

    def __init__(self,num_channel=32):   
        super().__init__()

        resnet18 = torch_models.resnet18()
        conv1 = resnet18.conv1
        bn1 = resnet18.bn1
        relu = resnet18.relu
        maxpool = resnet18.maxpool
        layer1 = resnet18.layer1
        layer2 = resnet18.layer2
        layer3 = resnet18.layer3
        layer4 = resnet18.layer4

        layer4[0].conv1 = nn.Conv2d(256,512,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
        layer4[0].downsample[0] = nn.Conv2d(256,512,kernel_size=(1,1),stride=(1,1),bias=False) 

        layer5 = nn.Sequential(
            nn.Conv2d(512,num_channel,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(num_channel))

        self.layers = nn.Sequential(conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,layer5)

        del resnet18

    def forward(self,inp):

        return self.layers(inp)



def proto_eval_k_shot(feature_vector,way,shot,dim):

    support = feature_vector[:way*shot].view(way,shot,dim)
    centroid = torch.mean(support,1).unsqueeze(0)
    query = feature_vector[way*shot:].unsqueeze(1)

    neg_l2_distance = torch.sum((centroid-query)**2,-1).neg().view(way*16,way)
    _,max_index = torch.max(neg_l2_distance,1)

    return max_index


def proto_forward_log_pred(feature_vector,train_shot,test_shot,dim,way):

    support = feature_vector[:way*train_shot].view(way,train_shot,dim)
    centroid = torch.mean(support,1).unsqueeze(0)
    query = feature_vector[way*train_shot:].unsqueeze(1)
        
    neg_l2_distance = torch.sum((centroid-query)**2,-1).neg().view(way*test_shot,way)
    log_prediction = F.log_softmax(neg_l2_distance,dim=1)

    return log_prediction



class Proto_Model(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False):
        
        super().__init__()
        if resnet:
            num_channel = 32
            self.feature_extractor = BackBone_ResNet(num_channel)
        else:
            num_channel = 64
            self.feature_extractor = BackBone(num_channel)
        self.shots = shots
        self.way = way
        self.num_channel = num_channel
        self.dim = num_channel

    
    def get_feature_vector(self,inp):
        pass

    
    def eval_k_shot(self,inp,way,shot):

        feature_vector = self.get_feature_vector(inp)        
        max_index = proto_eval_k_shot(feature_vector,
            way = way,
            shot = shot,
            dim = self.dim)

        return max_index


    def forward(self,inp):

        feature_vector = self.get_feature_vector(inp)        
        log_prediction = proto_forward_log_pred(feature_vector,
            train_shot = self.shots[0],
            test_shot = self.shots[1],
            dim = self.dim,
            way = self.way)

        return log_prediction



class PN_Model(nn.Module):
    
    def __init__(self,num_part,resnet=False):
        super().__init__()

        if resnet:

            num_channel = 32

            resnet18 = torch_models.resnet18()
            conv1 = resnet18.conv1
            bn1 = resnet18.bn1
            relu = resnet18.relu
            maxpool = resnet18.maxpool
            layer1 = resnet18.layer1
            layer2 = resnet18.layer2
            layer3 = resnet18.layer3
            layer4 = resnet18.layer4

            layer4[0].conv1 = nn.Conv2d(256,512,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
            layer4[0].downsample[0] = nn.Conv2d(256,512,kernel_size=(1,1),stride=(1,1),bias=False)

            layer5 = nn.Sequential(
                nn.Conv2d(512,num_channel,kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(num_channel))

            self.shared_layers = nn.Sequential(conv1,
                bn1,relu,maxpool,layer1,layer2,layer3)

            self.class_branch = nn.Sequential(layer4,layer5)

            self.part_branch = nn.Sequential(
                nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,num_part,kernel_size=3,stride=1,padding=1))

            del resnet18

        else:

            num_channel = 64

            self.shared_layers = nn.Sequential(
                ConvBlock(3,num_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                ConvBlock(num_channel,num_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

            self.part_branch = nn.Sequential(
                nn.Conv2d(num_channel,30,kernel_size=3,stride=2,padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30,num_part,kernel_size=3,stride=1,padding=1))

            self.class_branch = nn.Sequential(
                ConvBlock(num_channel,num_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                ConvBlock(num_channel,num_channel))

        self.num_channel = num_channel
        self.num_part = num_part
        self.dim = num_channel*num_part


    def get_heatmap(self,inp):

        logits = self.part_branch(self.shared_layers(inp))
        heat_map = nn.Sigmoid()(logits)
        return heat_map

    def eval_k_shot(self,inp,way,shot):

        feature_vector = self.get_feature_vector(inp)        
        max_index = proto_eval_k_shot(feature_vector,
            way = way,
            shot = shot,
            dim = self.dim)

        return max_index



class Dynamic_Model(nn.Module):
    
    def __init__(self,dim,num_class,way=None,shots=None,num_fake_novel_class=16):

        super().__init__()

        weight_base = torch.FloatTensor(num_class,dim).normal_(0.0, np.sqrt(2.0/(dim)))
        self.weight_base = nn.Parameter(weight_base,requires_grad=True)

        scale_cls = 10.0
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls),requires_grad=True)
        self.scale_cls_att = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls),requires_grad=True)

        self.phi_avg = nn.Parameter(torch.FloatTensor(dim).fill_(1),requires_grad=True)
        self.phi_att = nn.Parameter(torch.FloatTensor(dim).fill_(1),requires_grad=True)

        self.phi_q = nn.Linear(dim,dim)
        self.phi_q.weight.data.copy_(torch.eye(dim,dim)+torch.randn(dim,dim)*0.001)
        self.phi_q.bias.data.zero_()

        weight_keys = torch.FloatTensor(num_class,dim).normal_(0.0, np.sqrt(2.0/(dim)))
        self.weight_keys = nn.Parameter(weight_keys, requires_grad=True)

        self.dim = dim
        self.num_class = num_class
        self.num_fake_novel_class = num_fake_novel_class
        self.way = way
        self.shots = shots

    def get_feature_vector(self,inp):
        pass

    def weight_generator(self,fake_base_weight,fake_novel_feature_vector,fake_base_class_id):

        dim = self.dim
        num_fake_novel_class = self.num_fake_novel_class

        avg_feature_vector = torch.mean(fake_novel_feature_vector,dim=1) # 5class,channel
        avg_weight = self.phi_avg.unsqueeze(0)*avg_feature_vector # 5class,channel

        fake_base_weight = F.normalize(fake_base_weight,p=2,dim=1,eps=1e-12) # 155,channel
        
        query = self.phi_q(fake_novel_feature_vector.contiguous().view(num_fake_novel_class*5,dim)) # 25, channel
        query = F.normalize(query,p=2,dim=1,eps=1e-12) # 25,channel

        weight_keys = self.weight_keys[fake_base_class_id] # the keys of the base categoreis
        weight_keys = F.normalize(weight_keys,p=2,dim=1,eps=1e-12) # 155,channel

        logits = self.scale_cls_att*torch.matmul(query,weight_keys.transpose(0,1)) # 25,155
        att_score = F.softmax(logits,dim=1) # 25,155

        att_scored_fake_base_weight = torch.matmul(att_score,fake_base_weight) # 25,channel
        att_weight = self.phi_att*torch.mean(att_scored_fake_base_weight.view(num_fake_novel_class,5,dim),dim=1) # 5,channel

        fake_novel_weight = avg_weight+att_weight

        return fake_novel_weight

    def forward_stage_2(self,feature_vector,fake_novel_class_id,fake_base_class_id):
        
        dim = self.dim
        way = self.way
        shots = self.shots[0]
        weight_base = self.weight_base
        num_fake_novel_class = self.num_fake_novel_class

        feature_vector = feature_vector.view(way,shots,dim)
        feature_vector = F.normalize(feature_vector,p=2,dim=2,eps=1e-12)

        fake_novel_feature_vector = feature_vector[:num_fake_novel_class,:5,:] # 5 class,5 shot,channel
        
        fake_base_weight = weight_base[fake_base_class_id] # 155,channel
        fake_novel_weight = self.weight_generator(fake_base_weight,fake_novel_feature_vector,fake_base_class_id) # 5,channel

        weight_base_clone = weight_base.clone()
        weight_base_clone[fake_novel_class_id] = fake_novel_weight #160,channel

        feature_vector_test = feature_vector[:,5:,:].contiguous().view(way*15,dim)# 15,15,channel

        norm_weight = F.normalize(weight_base_clone,p=2,dim=1,eps=1e-12)
        
        logits = self.scale_cls*torch.matmul(feature_vector_test,norm_weight.transpose(0,1))

        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction

    def get_prediction(self,feature_vector,class_weight):

        batch_size = feature_vector.size(0)
        dim = self.dim
        
        feature_vector = feature_vector.view(batch_size,dim)
        feature_vector = F.normalize(feature_vector,p=2,dim=1,eps=1e-12)

        norm_weight = F.normalize(class_weight,p=2,dim=1,eps=1e-12)
        logits = self.scale_cls*torch.matmul(feature_vector,norm_weight.transpose(0,1))

        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction

    def get_single_class_weight(self,feature_vector):

        batch_size = feature_vector.size(0)
        dim = self.dim

        feature_vector = feature_vector.view(batch_size,dim)
        feature_vector = F.normalize(feature_vector,p=2,dim=1,eps=1e-12) # batch,channel

        avg_feature_vector = torch.mean(feature_vector,dim=0) # channel
        avg_weight = self.phi_avg*avg_feature_vector # channel

        norm_base_weight = F.normalize(self.weight_base,p=2,dim=1,eps=1e-12) # 160,channel
        
        query = self.phi_q(feature_vector) # batch, channel
        query = F.normalize(query,p=2,dim=1,eps=1e-12) # batch,channel

        weight_keys = self.weight_keys # the keys of the base categoreis
        weight_keys = F.normalize(weight_keys,p=2,dim=1,eps=1e-12) # 160,channel

        logits = self.scale_cls_att*torch.matmul(query,weight_keys.transpose(0,1)) # batch,160
        att_score = F.softmax(logits,dim=1) # batch,160

        att_scored_base_weight = torch.matmul(att_score,norm_base_weight) # batch,channel
        att_weight = self.phi_att*torch.mean(att_scored_base_weight,dim=0) # channel

        novel_weight = avg_weight+att_weight

        return novel_weight

    def eval_k_shot(self,feature_vector,way,shot):

        dim = self.dim
        support = feature_vector[:way*shot].view(way,shot,dim)
        class_weight = torch.zeros(way,dim).cuda()

        for i in range(way):
            class_weight[i] = self.get_single_class_weight(support[i])

        log_prediction = self.get_prediction(feature_vector[way*shot:],class_weight)
        _,max_index = torch.max(log_prediction,1)

        return max_index
