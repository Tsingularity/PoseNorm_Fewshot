import os
import torch
import argparse
from PIL import Image
import sys
sys.path.append('..')
from utils import util

parser = argparse.ArgumentParser()
parser.add_argument("--origin_path",help="directory of the original CUB dataset you download and extract",type=str)
args = parser.parse_args()

origin_path = args.origin_path
target_path = os.path.abspath('./cub_fewshot')
resolution = [84,224]

util.mkdir(target_path)

id2path = {}

with open(os.path.join(origin_path,'images.txt')) as f:
    lines = f.readlines()
    for line in lines:
        index, path = line.strip().split()
        index = int(index)
        id2path[index] = path

cat2name = {}

with open(os.path.join(origin_path,'classes.txt')) as f:
    lines = f.readlines()
    for line in lines:
        cat, name = line.strip().split()
        cat = int(cat)
        cat2name[cat] = name

cat2img = {}

with open(os.path.join(origin_path,'image_class_labels.txt')) as f:
    lines = f.readlines()
    for line in lines:
        image_id, class_id = line.strip().split()
        image_id = int(image_id)
        class_id = int(class_id)
        
        if class_id not in cat2img:
            cat2img[class_id]=[]
        cat2img[class_id].append(image_id)

support = []
val_ref = []
val_query = []
test_ref = []
test_query = []

support_cat = []
val_cat = []
test_cat = []

for i in range(1,201):
    img_list = cat2img[i]
    img_num = len(img_list)
    name = cat2name[i]

    if i%2 == 0:
        support_cat.append(name)
        support.extend(img_list)
    elif i%4 == 1:
        val_cat.append(name)
        val_ref.extend(img_list[:img_num//5])
        val_query.extend(img_list[img_num//5:])
    elif i%4 ==3:
        test_cat.append(name)
        test_ref.extend(img_list[:img_num//5])
        test_query.extend(img_list[img_num//5:])

id2bbx={}

with open(os.path.join(origin_path,'bounding_boxes.txt')) as f:
    lines = f.readlines()
    for line in lines:
        index,x,y,width,height = line.strip().split()
        index = int(index)
        x = float(x)
        y = float(y)
        width = float(width)
        height = float(height)
        id2bbx[index] = [x,y,width,height]

id2part={}

with open(os.path.join(origin_path,'parts','part_locs.txt')) as f:
    lines = f.readlines()
    for line in lines:
        index,part_id,x,y,visible = line.strip().split()
        index = int(index)
        x = float(x)
        y = float(y)
        visible = int(visible)
        if index not in id2part:
            id2part[index]=[]
        id2part[index].append([x,y,visible])

split = ['support','val/refer','val/query','test/refer','test/query']
split_cat = [support_cat,val_cat,test_cat]
split_img = [support,val_ref,val_query,test_ref,test_query]

for res in resolution:

    path2annot = {}
    
    res_dir = os.path.join(target_path,'res_'+str(res))
    util.mkdir(res_dir)
    
    for folder_name in ['support','val','test','val/refer','val/query','test/refer','test/query']:
        util.mkdir(os.path.join(res_dir,folder_name))
    
    for i in range(3):
        if i:
            for j in [2*i-1,2*i]:
                temp_path = os.path.join(res_dir,split[j])
                for cat_name in split_cat[i]: 
                    util.mkdir(os.path.join(temp_path,cat_name))
        else:
            temp_path = os.path.join(res_dir,split[i])
            for cat_name in split_cat[i]: 
                util.mkdir(os.path.join(temp_path,cat_name))

    for i in range(5):
        temp_path = os.path.join(res_dir,split[i])
        for index in split_img[i]:
            img_path = id2path[index]
            origin_img = os.path.join(origin_path,'images',img_path)
            target_img = os.path.join(temp_path,img_path[:-3]+'bmp')

            p = Image.open(origin_img)
            w,h = p.size
            p = p.resize((res,res),Image.BILINEAR)
            p.save(target_img)
        
            x,y,width,height = id2bbx[index]
            x_min = x/w
            x_max = (x+width)/w
            y_min = y/h
            y_max = (y+height)/h
            
            parts = id2part[index]
            new_parts = []
            for part in parts:
                x = part[0]/w
                y = part[1]/h
                new_parts.append([x,y,part[2]])

            path2annot[target_img] = {}
            path2annot[target_img]['bbx'] = [x_min,x_max,y_min,y_max]
            path2annot[target_img]['part'] = new_parts

    torch.save(path2annot,os.path.join(res_dir,'path2annot.pth'))


for res in resolution:
    
    res_dir = os.path.join(target_path,'res_%d'%(res))
    
    origin_dict = torch.load(os.path.join(res_dir,'path2annot.pth'))
    
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