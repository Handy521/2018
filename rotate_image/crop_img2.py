# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 22:36:19 2018

@author: yong2
"""
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU
anno_file = "pos-rot.txt"
pos_dir = "rorate2"
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print "%d pics in total" % num
idx = 0
p_idx = 0 
box_idx=0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = map(float, annotation[1:]) 
    boxes = np.array(bbox, dtype=np.float).reshape(-1, 4)
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print idx, "images done"
    height, width, channel = img.shape
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        
        # generate positive examples and part faces
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
            resized_im = cv2.resize(cropped_im, (32, 32), interpolation=cv2.INTER_LINEAR)

           # box_ = box.reshape(1, -1)
            
            box_ = box.reshape(1, -1)
            aa=IoU(crop_box, box_)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_dir, "%s.jpg"%p_idx)
                #f1.write("32pic/%s.jpg"%p_idx+" 1")
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
       # box_idx += 1
     #   print "%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx)

#f1.close()
#f2.close()
#f3.close()
