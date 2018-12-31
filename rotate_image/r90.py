# -*- coding: utf-8 -*-
"""
Created on Sat Oct 06 16:40:20 2018

@author: yong2
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:28:23 2018

@author: yong2
"""

import cv2
import numpy as np
import os

anno_file="pos-rot-new.txt"

dst_dir="r270"
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

g1 = open(os.path.join(dst_dir, 'p270.txt'), 'w')  
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print "%d pics in total" % num
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
a=[]
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = map(float, annotation[1:]) 
    #boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print idx, "images done"
    i=str(idx)
    height, width, channel = img.shape
#    t=cv2.transpose(img)
#    f=cv2.flip(t,0)
#    cv2.imwrite(os.path.join(dst_dir,i+'.jpg'),f)

 #   cv2.imwrite(os.path.join(dst_dir,i+'.jpg'),g)

    x1=bbox[1]
    y1=width-bbox[2]
    x2=bbox[3]
    y2=width-bbox[0]
    g1.write("r270\%s"%i+'.jpg' + ' %.2f %.2f %.2f %.2f\n'%(x1,y1,x2,y2))
 

g1.close()
   