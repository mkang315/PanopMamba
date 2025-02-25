import os
import numpy as np
import glob
import cv2
import scipy.io as sio
import skimage
import torch
from PIL import Image
import scipy
import scipy.ndimage
from skimage import measure
from torchmetrics.detection import PanopticQuality

from ioui import AccuracyMetric

def Panoptic_quality(ground_truth_image,predicted_image,ious):
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}
    #PQ_calculation(ground_truth_image,predicted_image)
    for i in np.unique(ground_truth_image):
        if i == 0:
            pass
        else:
            temp_image = np.array(ground_truth_image)
            temp_image = temp_image == i
            matched_image = temp_image * predicted_image

            for j in np.unique(matched_image):
                if j == 0:
                    pass
                else:
                    #pred_temp = predicted_image == j
                    # intersection = sum(sum(temp_image*pred_temp))
                    # union = sum(sum(temp_image + pred_temp))
                    IOU =  ious[i].item()
                    if IOU> 0.5:
                        matched_instances [i] = j, IOU

    pred_indx_list = np.unique(predicted_image)
    pred_indx_list = np.array(pred_indx_list[1:])

    # Loop on ground truth instances
    for indx in np.unique(ground_truth_image):
        if indx == 0:
            pass
        else:
            if indx in matched_instances.keys():
                pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == [indx][0]))
                TP = TP+1
                sum_IOU = sum_IOU+matched_instances[indx][1]
            else:
                FN = FN+1
    FP = len(np.unique(pred_indx_list))
    try:
        PQ = sum_IOU/(TP+0.5*FP+0.5*FN)
        SQ = sum_IOU/TP
        RQ = TP/(TP+0.5*FP+0.5*FN)
    except:
        PQ = 0
        SQ = 0
        RQ = 0

    return PQ,SQ,RQ
alltruth = [x for x in glob.glob(os.path.join("./outputs_txt/real/*.png"))  if x.find("_.png")<0]
alltruth = sorted(alltruth,key=lambda x: x.split(os.sep)[-1].split(".")[0])
allpred = [x for x in glob.glob(os.path.join("./outputs_txt/predict/*.png"))  if x.find("_.png")<0]
allpred = sorted(allpred,key=lambda x:  x.split(os.sep)[-1].split(".")[0])



label_dic={
        'Epithelial':1,
        'Lymphocyte':2,
        'Neutrophil':3,
        'Macrophage' :4
      }
i = 0

#  mIoUD
if False:

    allpqs = []
    allsq = []
    allrq = []
    allious =[]
    alpha = 0.01
    dist = cv2.DIST_L2
    metriccTotal = AccuracyMetric(num_classes=5)
    for truth,pred in zip(alltruth,allpred):
        gt_mask = skimage.io.imread(truth)
        pr_mask =  skimage.io.imread(pred)
        print(i,len(alltruth))
        metricc = AccuracyMetric(num_classes=5)
        tp, tn, fp, fn = metricc.add(pr_mask,gt_mask )
        metriccTotal.add(pr_mask,gt_mask )
        miouds= metricc.valueD()
        pa = Panoptic_quality(gt_mask,pr_mask,miouds)
        if float(pa[0]) !=0:
            allpqs.append(float(pa[0]))
        if float(pa[1]) !=0:
            allsq.append(float(pa[1]))
        if float(pa[2]) !=0:
            allrq.append(float(pa[2]))
        print(i,len(alltruth),pa)
        i +=1
    mIoUD = metriccTotal.valueD(isTotal=True)
i = 0
#  mIoUI
if True:
    allpqs = []
    allsq = []
    allrq = []
    allious =[]
    alpha = 0.01
    dist = cv2.DIST_L2
    metriccTotal = AccuracyMetric(num_classes=5)
    for truth,pred in zip(alltruth,allpred):
        gt_mask = skimage.io.imread(truth)
        pr_mask =  skimage.io.imread(pred)
        print(i,len(alltruth))
        metricc = AccuracyMetric(num_classes=5)
        tp, tn, fp, fn = metricc.add(pr_mask,gt_mask )
        metriccTotal.add(pr_mask,gt_mask )
        miouds= metricc.valueI()[0]
        pa = Panoptic_quality(gt_mask,pr_mask,miouds)
        if float(pa[0]) !=0:
            allpqs.append(float(pa[0]))
        if float(pa[1]) !=0:
            allsq.append(float(pa[1]))
        if float(pa[2]) !=0:
            allrq.append(float(pa[2]))
        print(i,len(alltruth),pa)
        i +=1
    mIoUD = metriccTotal.valueI(isTotal=True)
i = 0
#  mIoUIQ
if True:
    allpqs = []
    allsq = []
    allrq = []
    allious =[]
    alpha = 0.01
    dist = cv2.DIST_L2
    metriccTotal = AccuracyMetric(num_classes=5)
    for truth,pred in zip(alltruth,allpred):
        gt_mask = skimage.io.imread(truth)
        pr_mask =  skimage.io.imread(pred)
        print(i,len(alltruth))
        metricc = AccuracyMetric(num_classes=5)
        tp, tn, fp, fn = metricc.add(pr_mask,gt_mask )
        metriccTotal.add(pr_mask,gt_mask )
        miouds= metricc.valueI(isq=True)[0]
        pa = Panoptic_quality(gt_mask,pr_mask,miouds)
        if float(pa[0]) !=0:
            allpqs.append(float(pa[0]))
        if float(pa[1]) !=0:
            allsq.append(float(pa[1]))
        if float(pa[2]) !=0:
            allrq.append(float(pa[2]))
        print(i,len(alltruth),pa)
        i +=1
    mIoUD = metriccTotal.valueI(isq=True,isTotal=True)
i = 0
#  mIoUC
if True:
    allpqs = []
    allsq = []
    allrq = []
    allious =[]
    alpha = 0.01
    dist = cv2.DIST_L2
    metriccTotal = AccuracyMetric(num_classes=5)
    for truth,pred in zip(alltruth,allpred):
        gt_mask = skimage.io.imread(truth)
        pr_mask =  skimage.io.imread(pred)
        print(i,len(alltruth))
        metricc = AccuracyMetric(num_classes=5)
        tp, tn, fp, fn = metricc.add(pr_mask,gt_mask )
        metriccTotal.add(pr_mask,gt_mask )
        miouds= metricc.valueI()[0]
        pa = Panoptic_quality(gt_mask,pr_mask,miouds)
        if float(pa[0]) !=0:
            allpqs.append(float(pa[0]))
        if float(pa[1]) !=0:
            allsq.append(float(pa[1]))
        if float(pa[2]) !=0:
            allrq.append(float(pa[2]))
        print(i,len(alltruth),pa)
        i +=1
    mIoUD = metriccTotal.valueI(isTotal=True)
i = 0
#  mIoUCQ
if True:
    allpqs = []
    allsq = []
    allrq = []
    allious =[]
    alpha = 0.01
    dist = cv2.DIST_L2
    metriccTotal = AccuracyMetric(num_classes=5)
    for truth,pred in zip(alltruth,allpred):
        gt_mask = skimage.io.imread(truth)
        pr_mask =  skimage.io.imread(pred)
        print(i,len(alltruth))
        metricc = AccuracyMetric(num_classes=5)
        tp, tn, fp, fn = metricc.add(pr_mask,gt_mask )
        metriccTotal.add(pr_mask,gt_mask )
        miouds= metricc.valueI(isq=True)[0]
        pa = Panoptic_quality(gt_mask,pr_mask,miouds)
        if float(pa[0]) !=0:
            allpqs.append(float(pa[0]))
        if float(pa[1]) !=0:
            allsq.append(float(pa[1]))
        if float(pa[2]) !=0:
            allrq.append(float(pa[2]))
        print(i,len(alltruth),pa)
        i +=1
    mIoUD = metriccTotal.valueI(isq=True,isTotal=True)

print("mIoUD:",mIoUD.item())
print("PQD:",np.mean(np.array(allpqs)))
print("SQD:",np.mean(np.array(allsq)))
print("RQD:",np.mean(np.array(allrq)))

print("==============")
print("mIoUI:",mIoUD.item())
print("PQI:",np.mean(np.array(allpqs)))
print("SQI:",np.mean(np.array(allsq)))
print("RQI:",np.mean(np.array(allrq)))

print("==============")
print("mIoUIQ:",mIoUD.item())
print("PQIQ:",np.mean(np.array(allpqs)))
print("SQIQ:",np.mean(np.array(allsq)))
print("RQIQ:",np.mean(np.array(allrq)))

print("==============")
print("mIoUC:",mIoUD.item())
print("PQC:",np.mean(np.array(allpqs)))
print("SQC:",np.mean(np.array(allsq)))
print("RQC:",np.mean(np.array(allrq)))

print("==============")
print("mIoUCQ:",mIoUD.item())
print("PQCQ:",np.mean(np.array(allpqs)))
print("SQCQ:",np.mean(np.array(allsq)))
print("RQCQ:",np.mean(np.array(allrq)))