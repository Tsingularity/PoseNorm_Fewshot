import os
import torch
import math
import argparse
from PIL import Image
import numpy as np
import sys
sys.path.append('..')
from utils import util

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--origin_path",help="directory of the original FGVC dataset you download and extract",type=str)
args = parser.parse_args()

home_dir = os.path.join(args.origin_path,'data')
target_dir = os.path.abspath('./fgvc_fewshot')
resolution = [84,224]

util.mkdir(target_dir)

cat_id2name={}
cat_name2id={}
with open(os.path.join(home_dir,'variants.txt')) as f:
    content = f.readlines()
    for i in range(len(content)):
        name=content[i].strip()
        cat_id2name[i]=name
        cat_name2id[name]=i
        
img2cat={}
cat2img={}
for i in ['images_variant_trainval.txt','images_variant_test.txt']:
    with open(os.path.join(home_dir,i)) as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            img = line[:7]
            cat = line[8:]
            cat_id = cat_name2id[cat]
            img2cat[img] = cat_id
            if cat_id not in cat2img:
                cat2img[cat_id]=[]
            cat2img[cat_id].append(img)

img2bbx={}
with open(os.path.join(home_dir,'images_box.txt')) as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        img,xmin,ymin,xmax,ymax = line.split()
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)
        with Image.open(os.path.join(home_dir,'images',img+'.jpg')) as temp:
            width,height = temp.size
        height = height-20
        img2bbx[img] = [xmin/width,xmax/width,ymin/height,ymax/height]

support_cat=[]
val_cat=[]
test_cat=[]
for i in range(100):
    if i%2==0:
        support_cat.append(i)
    elif i%4==1:
        val_cat.append(i)
    elif i%4==3:
        test_cat.append(i)

for res in resolution:
    res_dir = os.path.join(target_dir,'res_'+str(res))
    util.mkdir(res_dir)
    for i in ['support','val','test','val/refer','val/query','test/refer','test/query']:
        util.mkdir(os.path.join(res_dir,i))
    dir_name = ['support','val/refer','val/query','test/refer','test/query']
    cat_list = [support_cat,val_cat,test_cat]
    for i in range(5):
        index = math.ceil(i/2)
        tar_cat_list = cat_list[index]
        for j in tar_cat_list:
            util.mkdir(os.path.join(res_dir,dir_name[i],str(j)))

def resize_img(path,res):
    img = Image.open(path)
    width = img.size[0]
    height = img.size[1]-20
    img = img.crop((0,0,width,height)).resize((res,res),Image.BILINEAR)
    return img


for res in resolution:
    
    path2annot = {}
    res_dir = os.path.join(target_dir,'res_'+str(res))
    for i in support_cat:
        for img in cat2img[i]:
            tar_img = resize_img(os.path.join(home_dir,'images',img+'.jpg'),res)
            target_path = os.path.join(res_dir,'support',str(i),img+'.bmp')
            tar_img.save(target_path)
            path2annot[target_path]={'bbx':img2bbx[img]}
    
    cat_list = [val_cat,test_cat]
    dir_name_1 = ['val','test']
    dir_name_2 = ['refer','query']
    
    for i in range(2):
        for j in cat_list[i]:
            img_list = cat2img[j]
            img_num = len(img_list)
            np.random.shuffle(img_list)
            
            refer_list = img_list[:img_num//5]
            query_list = img_list[img_num//5:]
            temp_list = [refer_list,query_list]
            for k in range(2):
                for img in temp_list[k]:
                    tar_img = resize_img(os.path.join(home_dir,'images',img+'.jpg'),res)
                    target_path = os.path.join(res_dir,dir_name_1[i],dir_name_2[k],str(j),img+'.bmp')
                    tar_img.save(target_path)
                    path2annot[target_path]={'bbx':img2bbx[img]}

    torch.save(path2annot,os.path.join(res_dir,'path2annot.pth'))

    ###eval k shot

    origin_dict = path2annot

    tar_dir = os.path.join(res_dir,'eval_k_shot')
    util.mkdir(tar_dir)

    cat_name = os.listdir(os.path.join(res_dir,'test/refer'))
    for cat in cat_name:
        util.mkdir(os.path.join(tar_dir,cat))

    tar_dict = {}

    for filename in origin_dict:
        
        if 'test/refer' in filename:
            tar_filename = filename.replace('test/refer','eval_k_shot')
        elif 'test/query' in filename:
            tar_filename = filename.replace('test/query','eval_k_shot')
        else:
            continue

        os.symlink(filename,tar_filename)

        tar_dict[tar_filename] = origin_dict[filename]

    torch.save(tar_dict,os.path.join(res_dir,'path2annot_eval_k_shot.pth'))