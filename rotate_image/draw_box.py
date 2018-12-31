# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:59:33 2018

@author: yong2
"""
import cv2
anno_file="r270/p270.txt"
with open(anno_file, 'r') as f:
    annotations = f.readlines()
aa=0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = map(float, annotation[1:]) 
    x1=bbox[0]
    y1=bbox[1]
    x2=bbox[2]
    y2=bbox[3]
    img = cv2.imread(im_path)
    cv2.rectangle(img, (int(x1+0.5),int(y1+0.5)), (int(x2+0.5),int(y2+0.5)), (0,255,0))
    cv2.namedWindow('test win', flags=0)  
    cv2.imshow('test win', img) 
    cv2.waitKey(0)  
    aa=aa+1
    print aa
#img = cv2.imread('1.jpg')
#cv2.rectangle(img, (244, 105),( 430, 277), (0,255,0))
#cv2.namedWindow('test win', flags=0)  
#cv2.imshow('test win', img) 
#cv2.waitKey(0)  