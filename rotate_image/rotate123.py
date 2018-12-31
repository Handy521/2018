# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:28:23 2018

@author: yong2
"""

import cv2
import numpy as np
import os
anno_file="train41.txt"
dst_dir="rorate270"
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print "%d pics in total" % num
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print idx, "images done"
    i=str(idx)
    t=cv2.transpose(img)
    f=cv2.flip(t,0)
    cv2.imwrite(os.path.join(dst_dir,i+'.jpg'),f)
#    g=cv2.flip(img,-1)
#    cv2.imwrite(os.path.join(dst_dir,i+'180.jpg'),g)


   