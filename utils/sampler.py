import os
import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler


class meta_batchsampler(Sampler):
    
    def __init__(self,data_source,way,shots):

        self.way = way
        self.shots = shots

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id


    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])       

        while len(temp_class2id) >= self.way:

            id_list = []

            list_class_id = list(temp_class2id.keys())

            pcount = np.array([len(temp_class2id[class_id]) for class_id in list_class_id])

            batch_class_id = np.random.choice(list_class_id,size=self.way,replace=False,p=pcount/sum(pcount))

            for shot in self.shots:
                for class_id in batch_class_id:
                    for _ in range(shot):
                        id_list.append(temp_class2id[class_id].pop())

            for class_id in batch_class_id:
                if len(temp_class2id[class_id])<sum(self.shots):
                    temp_class2id.pop(class_id)

            yield id_list


class ordered_sampler(Sampler):

    def __init__(self,data_source):

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id


    def __iter__(self):

        class2id = deepcopy(self.class2id)

        list_class_id = list(class2id.keys())

        for key in list_class_id:
            np.random.shuffle(class2id[key])

        for class_id in class2id:
            id_list = class2id[class_id]
            yield id_list


class random_sampler(Sampler):

    def __init__(self,data_source,way,shot,trial=600):

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id
        self.way = way
        self.shot = shot
        self.trial = trial
        self.test_shot = 16

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        test_shot = self.test_shot
        
        class2id = deepcopy(self.class2id)        
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []
 
            np.random.shuffle(list_class_id)
            
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])
                
            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot+test_shot)])

            yield id_list


class proto_less_annot_batchsampler(torch.utils.data.Sampler):
    
    def __init__(self,data_source,way,shots,percent,batch_size,seed=42):
        
        self.way = way
        self.shots = shots
        self.percent = percent

        iddict = dict()
        for _id,cat in enumerate(data_source.targets):
            if cat in iddict:
                iddict[cat].append(_id)
            else:
                iddict[cat] = [_id,]
        
        annodict = dict()
        np.random.seed(seed)
        
        for key in iddict.keys():
            
            L = len(iddict[key])
            newL = math.ceil(L*percent/100)
            np.random.shuffle(iddict[key])
            annodict[key] = iddict[key][:newL]
            
        self.iddict1 = annodict
        self.iddict2 = iddict
        self.batch_size = batch_size
    
    def __iter__(self):

        trackdict1 = deepcopy(self.iddict1)
        trackdict2 = deepcopy(self.iddict2)
        
        for key in trackdict1:
            np.random.shuffle(trackdict1[key])
            np.random.shuffle(trackdict2[key])
        
        while len(trackdict2.keys()) >= self.way:

            idlist = []
            
            pcount = np.array([len(trackdict2[k]) for k in list(trackdict2.keys())])
            cats = np.random.choice(list(trackdict2.keys()),size=self.way,replace=False,p=pcount/sum(pcount))
            
            for cat in cats:
                idlist.extend(np.random.choice(trackdict1[cat],size=self.batch_size,replace=False))
            
            for shot in self.shots:
                for cat in cats:
                    for _ in range(shot):
                        idlist.append(trackdict2[cat].pop())
            
            for cat in cats:
                if len(trackdict2[cat])<sum(self.shots):
                    trackdict2.pop(cat)
            
            yield idlist
