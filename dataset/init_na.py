from PIL import Image
import torch
import os
import numpy as np
import sys
import argparse
sys.path.append('..')
from utils import util

parser = argparse.ArgumentParser()
parser.add_argument("--origin_path",help="directory of the original nabirds dataset you download and extract",type=str)
args = parser.parse_args()

origin_path = args.origin_path
target_path = os.path.abspath('./na_fewshot')
exclude_na_id_list = torch.load('exclude_na_id_list.pth')
resolution = [84,224]

np.random.seed(42)
util.mkdir(target_path)

name2size={}

with open(os.path.join(origin_path,'sizes.txt')) as f:
    while True:
        content = f.readline().strip()
        if content == '':
            break
        content = content.split()
        name = content[0].replace('-','')
        width = int(content[1])
        height = int(content[2])
        name2size[name] = [width,height]

name2bbx={}

with open(os.path.join(origin_path,'bounding_boxes.txt')) as f:
    while True:
        content = f.readline().strip()
        if content == '':
            break
        content = content.split()
        name = content[0].replace('-','')
        x = int(content[1])
        y = int(content[2])
        width = int(content[3])
        height = int(content[4])
        
        [w,h] = name2size[name]
        x_min = x/w
        x_max = (x+width)/w
        y_min = y/h
        y_max = (y+height)/h
        name2bbx[name] = [x_min,x_max,y_min,y_max]

name2part={}

with open(os.path.join(origin_path,'parts/part_locs.txt')) as f:
    while True:
        content = f.readline().strip()
        if content == '':
            break
        content = content.split()
        name = content[0].replace('-','')
        x = int(content[2])
        y = int(content[3])
        visible = int(content[4])
        
        if name not in name2part:
            name2part[name] = []
        
        [w,h] = name2size[name]
        x = x/w
        y = y/h
        name2part[name].append([x,y,visible])

name2annotation = {}
for i in name2bbx:
    name2annotation[i] = {}
    name2annotation[i]['bbx'] = name2bbx[i]
    name2annotation[i]['part'] = name2part[i]

for res in resolution:

    res_dir = os.path.join(target_path,'res_'+str(res))
    util.mkdir(res_dir)
    util.mkdir(os.path.join(res_dir,'refer'))
    util.mkdir(os.path.join(res_dir,'query'))

    path2annot = {}

    for i in os.listdir(os.path.join(origin_path,'images')):

        if int(i) in exclude_na_id_list:
            continue
        util.mkdir(os.path.join(res_dir,'refer',i))
        util.mkdir(os.path.join(res_dir,'query',i))
        
        image_list = os.listdir(os.path.join(origin_path,'images',i))
        np.random.shuffle(image_list)
        
        num = len(image_list)
        refer_num = int(num/5)
        refer_list = image_list[:refer_num]
        query_list = image_list[refer_num:]
        
        img_list = [refer_list,query_list]
        folder_name = ['refer','query']

        for index in range(2):
            for j in img_list[index]:
                p = Image.open(os.path.join(origin_path,'images',i,j))
                p = p.convert('RGB')
                
                p = p.resize((res,res),Image.BILINEAR)
                target_img = os.path.join(res_dir,folder_name[index],i,j[:-3]+'bmp')
                p.save(target_img)
                path2annot[target_img] = name2annotation[j[:-4]]

    torch.save(path2annot,os.path.join(res_dir,'path2annot.pth'))