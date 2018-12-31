# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:22:43 2018

@author: yong2
"""

import numpy as np


import os
import cv2
import time
import h5py
import matplotlib.pyplot as plt
import matplotlib.lines as lines
def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr
def get_annotation(anno_path='anno.mat'):
    '''
    Parameters:
    ----------
    Returns:
    -------
    result:dict()
    e.g. {'1111.jpg':[np.array[x1,y1,x2,y2],np.array[x1,y1,x2,y2],}
    '''
    mat = h5py.File(anno_path, 'r')
    anno_list = mat[u'anno'][:]
    result=dict()
    for i in range(anno_list.shape[1]):

        file_name=''.join([chr(x) for x in mat[anno_list[0,i]][:]])
        bboxes=[]
        for j in range(mat[anno_list[1,i]].shape[0]):
            t=mat[anno_list[1,i]][j,0]
            bbox=mat[t][:].flatten('F')#(x1,y1,x2,y2)
            bboxes.append(bbox)
        result[file_name]=bboxes
    return result
def get_detection(detetion_path):
    """
    Parameters:
    ----------
    detetion_path: detection file
        file_name
        face_num
        <x1,y1,x2,y2,score>

        e.g.
        1004109301.jpg
        1
        225.928930283 241.036109924 431.988132477 515.827237129 1.000
    Returns:
    -------
    detection:[{'file_name':file_name,"bboxes":bboxes},]
              bboxes: N*np.array([x1,y1,x2,y2,score])
    """ 
    detection=[]
    f_detection=open(detetion_path,'r')
    while True:
        line = f_detection.readline()
        if not line:
            break
        file_name=line.strip()
        face_num = int(f_detection.readline().strip())
        bboxes=[]
        for i in xrange(face_num):
            line=f_detection.readline()
            bbox=line.strip().split()
            #major_axis_radius minor_axis_radius angle center_x center_y 1
            bbox=np.array([float(bbox[x]) for x in range(5)])
            bboxes.append(bbox)
        detection.append({'file_name':file_name,"bboxes":bboxes})
    return detection 



def voc_ap(rec, prec, use_07_metric=False):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(annotaions,detections,ovrthresh=0.5):
    #assert len(annotations)==len(detections)
    detections_unfolded=[]
    for detection in detections:
        for bbox in detection["bboxes"]:
            detections_unfolded.append({"file_name":detection['file_name'],"bbox":bbox[:4],"score":bbox[4]})
    detections_unfolded.sort(key=lambda x:x["score"])
    detections_unfolded=detections_unfolded[::-1]
    # go down dets and mark TPs and FPs
    nd=len(detections_unfolded)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    d=0
    for detection  in detections_unfolded:
        if(d==153):
            print d
        if(len(annotations[detection['file_name']])!=0):#若某文件的真实区域还有未与检测狂匹配的，可以进行下一步
            ovr=IoU(detection['bbox'],np.array(annotations[detection['file_name']]))
            ovrmax_ind=np.argmax(ovr)#与检测结果重叠最大的真实区域下标
            ovrmax=np.max(ovr)
            if ovrmax > ovrthresh:#大于阈值则认为检测的结果正确
                tp[d] = 1.
                annotations[detection['file_name']].pop(ovrmax_ind)#把真实区域删去

            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
        d+=1

    # compute precision recall#采用累积和
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    npos=sum([len(v) for k,v in annotations.items() ])#未检测到的
    rec = tp /852 #852召回率=tp/(tp+fn)=tp/真实区域的数目
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap


if __name__ == '__main__':
    #annotations=get_annotation('anno.mat')
    path1='rotface_output2.txt'
    #path1='fold_rot-out.txt'
    detections=get_detection(path1)
    annotation=get_detection('rotface_annotation2.txt')
    annotations=dict()
    for annota in annotation:
        file_name=annota['file_name']
        bboxes=annota['bboxes']
        annotations[file_name]=bboxes
    rec, prec, ap=voc_eval(annotations,detections)
    plt.plot(rec,prec )
    plt.plot(rec1,prec1 )
    plt.axis([0,0.85,0,1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR curve')
    plt.show()
