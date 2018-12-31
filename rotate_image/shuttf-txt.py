# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 11:35:52 2018

@author: yong2
"""

import numpy as np
anno_file="train012.txt"
g1 = open( 'shuttf-train0123.txt', 'w')  
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
random_num=np.random.permutation(num)

for ii in range(num):
    a=random_num[ii]
    g1.write(annotations[a])
g1.close()