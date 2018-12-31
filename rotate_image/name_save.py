# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 23:17:04 2018

@author: yong2
"""

import numpy as np
import os

dir1="rorate270"
list1=os.listdir(dir1)
f1 = open(('train3.txt'), 'w') 

for i in range(0,len(list1)):
    path=os.path.join(dir1,list1[i])
    
    
    f1.write(str(path)+" 3"+"\n")
  
f1.close()