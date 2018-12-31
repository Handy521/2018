# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:35:33 2018

@author: yong2
"""
import os
import xml.etree.ElementTree as ET
anno="xml"
list1=os.listdir(anno)
f1 = open(('anno_rot.txt'), 'w') 

for i in range(0,len(list1)):
    pathxml=os.path.join(anno,list1[i])
    tree=ET.parse(pathxml)   
    root = tree.getroot()
    path = root.find('path').text
    f1.write(str(path)[3:-4]+"\n")
    j=0
    for obj in root.iter('object'):                   
        j=j+1  
    f1.write(str(j)+'\n')    
    for obj in root.iter('object'):        
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))        
        f1.write(" ".join([str(a) for a in b]))
        f1.write(' 1'+'\n')
    
f1.close()