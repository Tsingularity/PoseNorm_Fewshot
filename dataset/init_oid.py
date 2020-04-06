import os
import numpy as np
from PIL import Image,ImageDraw
import scipy.io as scio
import argparse
import sys
sys.path.append('..')
from utils import util

parser = argparse.ArgumentParser()
parser.add_argument("--oid_origin_path",help="directory of the original OID dataset you download and extract",type=str)
parser.add_argument("--fgvc_origin_path",help="directory of the original FGVC dataset you download and extract",type=str)
args = parser.parse_args()

origin_path = os.path.join(args.oid_origin_path,'data/images/aeroplane')
target_path = './oid_fewshot'
resolution = [84,224]
feature_map = [10,14]

util.mkdir(target_path)

fg_list = os.listdir(os.path.join(args.fgvc_origin_path,'data/images'))
mat_file = os.path.join(args.oid_origin_path,'data/annotations/anno.mat')

data = scio.loadmat(mat_file,struct_as_record=False, squeeze_me=True)
anno = data['anno']

path2id={}
for i in range(len(anno.aeroplane.id)):
    path=anno.image.name[anno.aeroplane.parentId[i]-1]
    if path not in path2id:
        path2id[path]=[]
    path2id[path].append(anno.aeroplane.id[i])

id2part={}
for i in range(len(anno.aeroplane.id)):
    id2part[anno.aeroplane.id[i]]={}
    id2part[anno.aeroplane.id[i]]['aero']=[anno.aeroplane.polygon[i]]

for i in range(len(anno.wing.id)):
    par_id = anno.wing.parentId[i]
    if 'wing' not in id2part[par_id]:
        id2part[par_id]['wing']=[]
    id2part[par_id]['wing'].append(anno.wing.polygon[i])

for i in range(len(anno.wheel.id)):
    par_id = anno.wheel.parentId[i]
    if 'wheel' not in id2part[par_id]:
        id2part[par_id]['wheel']=[]
    id2part[par_id]['wheel'].append(anno.wheel.polygon[i])

for i in range(len(anno.verticalStabilizer.id)):
    par_id = anno.verticalStabilizer.parentId[i]
    if 'vertical' not in id2part[par_id]:
        id2part[par_id]['vertical']=[]
    id2part[par_id]['vertical'].append(anno.verticalStabilizer.polygon[i])

for i in range(len(anno.nose.id)):
    par_id = anno.nose.parentId[i]
    if 'nose' not in id2part[par_id]:
        id2part[par_id]['nose']=[]
    id2part[par_id]['nose'].append(anno.wheel.polygon[i])

valid_path = []
for i in list(path2id.keys()):
    if i not in fg_list:
        if len(path2id[i])==1:
            valid_path.append(i)


for i in range(2):
    
    res = resolution[i]
    fm_size = feature_map[i]
    par_dir = os.path.join(target_path,'res_%d'%(res))
    util.mkdir(par_dir)
    
    for name in ['origin','aero','wing','wheel','vertical','nose']:
        util.mkdir(os.path.join(par_dir,name))
    
    for j in valid_path:
        origin_img = Image.open(os.path.join(origin_path,j))
        width = origin_img.size[0]
        height = origin_img.size[1]-20
        tar_img = origin_img.crop((0,0,width,height)).resize((res,res),Image.BILINEAR)
        tar_img.save(os.path.join(par_dir,'origin',j[:-3]+'bmp'))
        
        _id = path2id[j][0]
            
        scalar = np.array([[fm_size/width],[fm_size/height]])
        for part in ['aero','wing','wheel','vertical','nose']:
            temp_img=Image.new('L',(fm_size,fm_size),0)        
            if part in id2part[_id]:
                for poly in id2part[_id][part]:
                    my_poly = poly*scalar
                    my_poly = my_poly.T.flatten().tolist()
                    ImageDraw.Draw(temp_img).polygon(my_poly,outline=255,fill=255)
            temp_img.save(os.path.join(par_dir,part,j[:-3]+'bmp'))


