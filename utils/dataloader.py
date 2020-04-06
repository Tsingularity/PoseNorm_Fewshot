import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image
from . import sampler


mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)])


def get_bird_dataset(data_path,annot,annot_path,flip):

    annot_dict = None
    if annot is not None:
        annot_dict = torch.load(annot_path)

    dataset = datasets.ImageFolder(
        data_path,
        loader = lambda x: bird_loader(path=x,flip=flip,
            annot_dict=annot_dict,annot=annot))

    return dataset



def meta_train_dataloader(data_path,shots,way,annot=None,annot_path=None):

    dataset = get_bird_dataset(data_path,annot=annot,annot_path=annot_path,flip=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = sampler.meta_batchsampler(data_source=dataset,way=way,shots=shots),
        num_workers = 3,
        pin_memory = False)

    return loader



def eval_dataloader(data_path,annot=None,annot_path=None):

    dataset = get_bird_dataset(data_path,annot=annot,annot_path=annot_path,flip=False)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = sampler.ordered_sampler(data_source=dataset),
        num_workers = 3,
        pin_memory = False)

    return loader



def eval_k_shot_dataloader(data_path,way,shot,annot=None,annot_path=None):

    dataset = get_bird_dataset(data_path,annot=annot,annot_path=annot_path,flip=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = sampler.random_sampler(data_source=dataset,way=way,shot=shot),
        num_workers = 3,
        pin_memory = False)

    return loader


def normal_train_dataloader(data_path,batch_size,annot=None,annot_path=None):
    
    dataset = get_bird_dataset(data_path,annot=annot,annot_path=annot_path,flip=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 3,
        pin_memory = False,
        drop_last=True)

    return loader


def oid_dataloader(data_path,batch_size,flip=True):

    oid_dataset = OidDataset(data_path,flip)
    
    dataloader = torch.utils.data.DataLoader(oid_dataset,
        batch_size=batch_size,shuffle=True,num_workers=5)

    return dataloader



def proto_train_less_annot_dataloader(data_path,shots,way,
    percent,annot_path,batch_size):

    dataset = get_bird_dataset(data_path,annot='part',annot_path=annot_path,flip=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = sampler.proto_less_annot_batchsampler(data_source=dataset,
            way=way,shots=shots,percent=percent,batch_size=batch_size),
        num_workers = 3,
        pin_memory = False)

    return loader



def bird_loader(path,flip=False,annot=None,annot_dict=None):

    p = Image.open(path)

    flip = flip and np.random.choice([True,False])

    if flip:
        p = p.transpose(Image.FLIP_LEFT_RIGHT)

    p = p.convert('RGB')

    p = transform(p)

    if annot is None:
        return p

    else:
        
        p_size = p.size(-1)
        if p_size == 224:
            fm_size = 14
        elif p_size == 84:
            fm_size = 10

        if annot=='bbx':

            mask = np.zeros((fm_size,fm_size))

            box = annot_dict[path]['bbx']

            for i in range(4):
                if box[i]>1:
                    box[i]=1

            x_min = fm_size*box[0]
            x_max = fm_size*box[1]
            y_min = fm_size*box[2]
            y_max = fm_size*box[3]

            x_min_int = int(x_min)
            x_max_int = int(x_max-0.0000001)+1
            y_min_int = int(y_min)
            y_max_int = int(y_max-0.0000001)+1

            if flip:

                mask[y_min_int:y_max_int,fm_size-x_max_int:fm_size-x_min_int] = 1

                # fade out
                mask[:, fm_size-x_min_int-1] *= 1-(x_min-x_min_int)
                mask[:, fm_size-x_max_int] *= 1-(x_max_int-x_max)

            else:

                mask[y_min_int:y_max_int,x_min_int:x_max_int] = 1

                # fade out
                mask[:,x_min_int] *= 1-(x_min-x_min_int)
                mask[:,x_max_int-1] *= 1-(x_max_int-x_max)

            mask[y_min_int,:] *= 1-(y_min-y_min_int)

            mask[y_max_int-1,:] *= 1-(y_max_int-y_max)

            mask = torch.FloatTensor(mask).unsqueeze(0)

            return [p,mask]

        elif annot=='part':

            num_part = 15

            mask = np.zeros((num_part,fm_size,fm_size))

            part_loc = np.array(annot_dict[path]['part'])

            if flip:

                part_loc[[6,10]] = part_loc[[10,6]]
                part_loc[[7,11]] = part_loc[[11,7]]
                part_loc[[8,12]] = part_loc[[12,8]]

            for i in range(15):

                if part_loc[i][2]==0:
                    continue

                if part_loc[i][0]>=1:
                    part_loc[i][0]=0.99999999
                if part_loc[i][1]>=1:
                    part_loc[i][1]=0.99999999

                x_int = int(fm_size*part_loc[i][0])
                y_int = int(fm_size*part_loc[i][1])

                if flip:
                    mask[i][y_int][fm_size-1-x_int] = 1

                else:
                    mask[i][y_int][x_int] = 1

            mask = torch.FloatTensor(mask)

            return [p,mask]



class OidDataset(torch.utils.data.Dataset):

    def __init__(self,root_dir,flip=True):

        img_list = os.listdir(os.path.join(root_dir,'origin'))
        length = len(img_list)

        self.length = length
        self.img_list = img_list
        self.flip = flip
        self.root_dir = root_dir

    def __len__(self):
        return self.length

    def __getitem__(self,idx):

        img_list = self.img_list
        root_dir = self.root_dir

        flip = np.random.choice([True,False]) and self.flip
        origin_img = Image.open(os.path.join(root_dir,'origin',img_list[idx]))
        if flip:
            origin_img = origin_img.transpose(Image.FLIP_LEFT_RIGHT)
        origin_img = origin_img.convert('RGB')
        img_tensor = transform(origin_img)
        
        part_arr = []
        for part in ['aero','wing','wheel','vertical','nose']:
            part_img = Image.open(os.path.join(root_dir,part,img_list[idx]))
            if flip:
                part_img = part_img.transpose(Image.FLIP_LEFT_RIGHT)
            part_arr.append(np.array(part_img))
        part_arr = np.stack(part_arr,axis=0)/255

        part_tensor = torch.FloatTensor(part_arr)

        return [img_tensor,part_tensor]